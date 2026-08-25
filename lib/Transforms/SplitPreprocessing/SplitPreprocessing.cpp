#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"      // from @llvm-project

#define DEBUG_TYPE "split-preprocessing"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SPLITPREPROCESSING
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h.inc"

using func::FuncOp;

namespace {

// Walk the IR to collect the set of operations that need to be cloned, upstream
// of a set of Encode ops, including recursing into regions and parents.
static SetVector<Operation*> computeDependenciesToClone(
    ArrayRef<Operation*> encodeOps, FuncOp containingFunc) {
  assert(containingFunc && "Boundary containingFunc must not be null");
  SetVector<Operation*> opsToClone;
  SmallVector<Operation*> worklist;

  // Dedupe when adding to the worklist.
  auto pushAsNeeded = [&](Operation* op) {
    if (op && op != containingFunc.getOperation() && opsToClone.insert(op)) {
      worklist.push_back(op);
    }
  };

  // Seed worklist
  for (Operation* op : encodeOps) {
    pushAsNeeded(op);
  }

  while (!worklist.empty()) {
    Operation* current = worklist.pop_back_val();

    // Preserve enclosing control flow (up to FuncOp)
    pushAsNeeded(current->getParentOp());

    // If the current operation contains regions, push their terminators
    // to propagate to the containing region.
    for (Region& region : current->getRegions()) {
      for (Block& block : region) {
        if (Operation* terminator = block.getTerminator()) {
          pushAsNeeded(terminator);
        }
      }
    }

    // Trace backward through operands
    for (Value operand : current->getOperands()) {
      if (Operation* definingOp = operand.getDefiningOp()) {
        pushAsNeeded(definingOp);
      } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        pushAsNeeded(blockArg.getOwner()->getParentOp());
      }
    }
  }

  return opsToClone;
}

struct PreprocessingAnalysis {
  SmallVector<Operation*> encodeOps;
  SetVector<Value> inputs;
  SetVector<Operation*> opsToClone;
};

// Encode ops can be elementwise-mappable, so if we're encoding a tensor we need
// to extract and store each element separately. This is unsupported for now
// because the storage memref analysis assumes each encode op corresponds to a
// single plaintext, but a tensor<1x!pt> is OK.
static bool isAllowedPlaintextType(Type type) {
  if (isa<PlaintextTypeInterface>(type)) return true;
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    auto shape = shapedTy.getShape();
    return shape.size() == 1 && shape[0] == 1;
  }
  return false;
}

// Rebuild each affine.for in `funcOp` without the iter_args whose region
// argument and loop result are both unused.
//
// When cloning a loop into the preprocessing function we leave behind a dummy
// initializer for any loop-carried value that isn't part of the plaintext
// slice (e.g. a ciphertext accumulator), relying on remove-dead-values to drop
// the now-unused iter_arg. That works for scf.for, but remove-dead-values only
// poisons (it does not structurally remove) dead affine.for iter_args, so a
// ub.poison-typed loop-carried value would otherwise survive into backend
// lowering, where it cannot be converted or emitted. This performs the removal
// remove-dead-values cannot, after which the dummy initializers become dead and
// are cleaned up normally.
static void removeDeadAffineForIterArgs(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());

  SmallVector<affine::AffineForOp> loops;
  funcOp.walk([&](affine::AffineForOp forOp) { loops.push_back(forOp); });

  for (affine::AffineForOp forOp : loops) {
    unsigned numIterArgs = forOp.getNumIterOperands();
    if (numIterArgs == 0) continue;

    SmallVector<unsigned> keptIndices;
    for (unsigned i = 0; i < numIterArgs; ++i) {
      if (!forOp.getRegionIterArgs()[i].use_empty() ||
          !forOp.getResult(i).use_empty()) {
        keptIndices.push_back(i);
      }
    }
    if (keptIndices.size() == numIterArgs) continue;  // nothing dead

    rewriter.setInsertionPoint(forOp);
    SmallVector<Value> keptInits;
    for (unsigned i : keptIndices) keptInits.push_back(forOp.getInits()[i]);

    auto newLoop = affine::AffineForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBoundOperands(),
        forOp.getLowerBoundMap(), forOp.getUpperBoundOperands(),
        forOp.getUpperBoundMap(), forOp.getStepAsInt(), keptInits);

    // Trim the existing terminator down to the kept loop-carried values.
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> keptYields;
    for (unsigned i : keptIndices) keptYields.push_back(yieldOp.getOperand(i));
    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldOp.getOperandsMutable().assign(keptYields); });

    // With no kept iter_args the builder added a default terminator; drop it so
    // the merged (trimmed) affine.yield is the loop's only terminator.
    if (keptInits.empty()) {
      rewriter.eraseOp(newLoop.getBody()->getTerminator());
    }

    // Map the old block arguments onto the new loop: induction var, then each
    // iter_arg. Kept ones map to the new region args; dead ones are unused, so
    // their (type-matched, dominating) original initializer is a safe
    // placeholder that is never actually referenced.
    SmallVector<Value> blockArgReplacements;
    blockArgReplacements.push_back(newLoop.getInductionVar());
    unsigned keptCursor = 0;
    for (unsigned i = 0; i < numIterArgs; ++i) {
      if (keptCursor < keptIndices.size() && keptIndices[keptCursor] == i) {
        blockArgReplacements.push_back(
            newLoop.getRegionIterArgs()[keptCursor++]);
      } else {
        blockArgReplacements.push_back(forOp.getInits()[i]);
      }
    }
    rewriter.mergeBlocks(forOp.getBody(), newLoop.getBody(),
                         blockArgReplacements);

    for (auto [newIdx, oldIdx] : llvm::enumerate(keptIndices)) {
      rewriter.replaceAllUsesWith(forOp.getResult(oldIdx),
                                  newLoop.getResult(newIdx));
    }
    rewriter.eraseOp(forOp);
  }
}

struct SplitPreprocessingPass
    : impl::SplitPreprocessingBase<SplitPreprocessingPass> {
  using SplitPreprocessingBase::SplitPreprocessingBase;

  void runOnOperation() override {
    Operation* root = getOperation();

    // The storage layout sizes each site by the enclosing loops' trip counts,
    // while the store/load indices are those loops' induction variables.
    // Normalize affine loops before capturing those indices so a later loop
    // normalization cannot rewrite a captured index to an affine.apply of the
    // original non-zero-based, non-unit-step induction variable.
    OpPassManager normalizeLoops("builtin.module");
    normalizeLoops.addNestedPass<func::FuncOp>(
        affine::createAffineLoopNormalizePass(true));
    if (failed(runPipeline(normalizeLoops, root))) {
      signalPassFailure();
      return;
    }

    // Annotate each encode op with a stable site id
    int32_t encodeId = 0;
    root->walk([&](PlaintextEncodeOpInterface op) {
      op->setAttr(
          "split_preprocessing_site_id",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), encodeId++));
    });

    root->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
      if (op.isDeclaration() || isClientHelper(op)) {
        return;
      }
      convertFunc(op);
    });

    clearAttrs(root, "split_preprocessing_site_id");
  }

  void convertFunc(FuncOp funcOp) {
    PreprocessingAnalysis analysis = analyzePreprocessing(funcOp);
    if (analysis.encodeOps.empty()) {
      return;
    }

    FailureOr<FuncOp> maybePreprocessingFuncOp =
        createPreprocessingFunction(funcOp, analysis);
    if (failed(maybePreprocessingFuncOp)) {
      signalPassFailure();
      return;
    }
    FuncOp preprocessingFuncOp = maybePreprocessingFuncOp.value();

    FailureOr<FuncOp> maybePreprocessedFuncOp =
        createPreprocessedFunction(funcOp, preprocessingFuncOp, analysis);
    if (failed(maybePreprocessedFuncOp)) {
      preprocessingFuncOp.erase();
      signalPassFailure();
      return;
    }
    FuncOp preprocessedFuncOp = maybePreprocessedFuncOp.value();

    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    moduleOp.insert(funcOp.getOperation(), preprocessingFuncOp);
    moduleOp.insert(funcOp.getOperation(), preprocessedFuncOp);

    updateOriginalFunc(funcOp, preprocessingFuncOp, preprocessedFuncOp,
                       analysis);

    // Remove dead values to clean up the created/updated functions
    OpPassManager pipeline("func.func");
    pipeline.addPass(createRemoveDeadValuesPass());
    (void)runPipeline(pipeline, preprocessingFuncOp);
    (void)runPipeline(pipeline, preprocessedFuncOp);
    (void)runPipeline(pipeline, funcOp);

    // remove-dead-values poisons but cannot structurally strip a dead
    // affine.for iter_arg (it only does so for scf.for), so the dummy
    // ciphertext iter_arg left when cloning a loop survives as a ub.poison
    // loop-carried value that later backend lowering can neither convert nor
    // emit. Strip those dead iter_args now, then re-run the cleanup so the
    // orphaned ub.poison initializers are removed too.
    removeDeadAffineForIterArgs(preprocessingFuncOp);
    (void)runPipeline(pipeline, preprocessingFuncOp);
  }

  void updateOriginalFunc(FuncOp funcOp, FuncOp preprocessingFuncOp,
                          FuncOp preprocessedFuncOp,
                          const PreprocessingAnalysis& analysis) {
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    auto preprocessingCall =
        func::CallOp::create(builder, funcOp.getLoc(), preprocessingFuncOp,
                             llvm::to_vector(analysis.inputs));

    SmallVector<Value> preprocessedArgs(funcOp.getArguments().begin(),
                                        funcOp.getArguments().end());
    preprocessedArgs.push_back(preprocessingCall.getResult(0));
    auto preprocessedCall = func::CallOp::create(
        builder, funcOp.getLoc(), preprocessedFuncOp, preprocessedArgs);

    Block* originalEntry = &funcOp.getBody().front();
    Operation* originalTerminator = originalEntry->getTerminator();
    originalTerminator->setOperands(preprocessedCall.getResults());

    DenseSet<Operation*> opsToKeep = {preprocessingCall, preprocessedCall,
                                      originalTerminator};
    for (Operation& op : llvm::make_early_inc_range(
             llvm::reverse(originalEntry->getOperations()))) {
      if (!opsToKeep.contains(&op)) op.erase();
    }
  }

  // Recursively clone an op and all ops in nested regions that are in the
  // opsToClone filter.
  static Operation* recursiveCloneOpWithFilter(
      Operation* srcOp, IRMapping& mapper,
      const SetVector<Operation*>& opsToClone, OpBuilder& builder) {
    // 1. Clone the operation shell without its nested regions
    Operation* targetOp = builder.cloneWithoutRegions(*srcOp, mapper);

    // 2. Recursively populate all blocks and regions
    for (auto [srcRegion, targetRegion] :
         llvm::zip(srcOp->getRegions(), targetOp->getRegions())) {
      for (Block& srcBlock : srcRegion) {
        // Append a new block to the target region
        Block* targetBlock = builder.createBlock(&targetRegion);
        // Map block arguments
        for (BlockArgument arg : srcBlock.getArguments()) {
          targetBlock->addArgument(arg.getType(), arg.getLoc());
          mapper.map(arg, targetBlock->getArguments().back());
        }

        // Selectively clone all operations in this block that belong to our
        // live slice
        for (Operation& childOp : srcBlock) {
          if (opsToClone.contains(&childOp)) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(targetBlock);
            recursiveCloneOpWithFilter(&childOp, mapper, opsToClone, builder);
          }
        }
      }
    }
    return targetOp;
  }

  // Get the set of induction variable indices in a loop nest surrounding an
  // encode op.
  SmallVector<Value> getContextualLoopIndices(Operation* encodeOp,
                                              Operation* maxParent) {
    SmallVector<Value> indices;
    Operation* parent = encodeOp->getParentOp();
    while (parent && parent != maxParent) {
      if (auto loopLikeOp = dyn_cast<LoopLikeOpInterface>(parent)) {
        auto inductionVar = loopLikeOp.getSingleInductionVar();
        if (inductionVar.has_value()) indices.push_back(*inductionVar);
      }
      parent = parent->getParentOp();
    }
    std::reverse(indices.begin(), indices.end());
    return indices;
  }

  FailureOr<FuncOp> createPreprocessingFunction(
      FuncOp op, const PreprocessingAnalysis& analysis) {
    MLIRContext* context = op.getContext();
    OpBuilder builder(context);

    SmallVector<Type> newInputs;
    newInputs.reserve(analysis.inputs.size());
    for (const auto& input : analysis.inputs) {
      newInputs.push_back(input.getType());
    }

    // Create a preprocessing.storage type with the set plaintext types
    // Preserve first-occurrence order as it determines storage field indices.
    SmallVector<Type> encodeTypes;
    for (Operation* input : analysis.encodeOps) {
      Type encodeTy = getElementTypeOrSelf(input->getResult(0).getType());
      if (!llvm::is_contained(encodeTypes, encodeTy))
        encodeTypes.push_back(encodeTy);
    }
    auto storageTy =
        preprocessing::PreprocessingStorageType::get(context, encodeTypes);

    // Record where each preprocessing parameter comes from, so a consumer does
    // not have to recover it from the call site in the combined entry func.
    SmallVector<int64_t> entryArgIndices;
    entryArgIndices.reserve(analysis.inputs.size());
    for (const auto& input : analysis.inputs) {
      auto blockArg = dyn_cast<BlockArgument>(input);
      entryArgIndices.push_back(blockArg && blockArg.getOwner() ==
                                                &op.getBody().front()
                                    ? blockArg.getArgNumber()
                                    : -1);
    }

    // Create the new func and annotate it appropriately
    auto funcType = FunctionType::get(context, newInputs, {storageTy});
    auto funcName = op.getName().str() + "__preprocessing";
    auto funcOp = FuncOp::create(op.getLoc(), funcName, funcType);
    funcOp.setVisibility(op.getVisibility());
    funcOp->setAttr(
        kServerPreprocessingFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(op.getName())),
            builder.getNamedAttr(kServerPreprocessingEntryArgs,
                                 builder.getDenseI64ArrayAttr(entryArgIndices)),
        }));

    // Set up the operation cloning infra: map the analysis-identified inputs to
    // the new func's block arguments
    IRMapping map;
    Block* entryBlock = funcOp.addEntryBlock();
    for (auto [idx, input] : llvm::enumerate(analysis.inputs)) {
      map.map(input, entryBlock->getArgument(idx));
    }

    builder.setInsertionPointToEnd(entryBlock);
    auto emptyOp =
        preprocessing::EmptyOp::create(builder, funcOp.getLoc(), storageTy);
    Value storage = emptyOp.getStorage();

    // When cloning a loop to the preprocessing function, the loop may have a
    // ciphertext iter_arg, whose initializer and other upstream ops we don't
    // want to clone. However, the op still needs a valid SSA value for its
    // initializer. In this case, we materialize an op of the right type
    // via UnrealizedConversionCast (which can create an op of any type from
    // nothing), and later allow remove-dead-values to clean up the iter_arg,
    // since it will be naturally unused.
    for (Operation* opToClone : analysis.opsToClone) {
      for (Value operand : opToClone->getOperands()) {
        if (!map.contains(operand) &&
            !analysis.opsToClone.contains(operand.getDefiningOp())) {
          if (isa<BlockArgument>(operand) &&
              analysis.opsToClone.contains(
                  cast<BlockArgument>(operand).getOwner()->getParentOp())) {
            continue;
          }
          Value dummy =
              UnrealizedConversionCastOp::create(
                  builder, funcOp.getLoc(), operand.getType(), ValueRange{})
                  .getResult(0);
          map.map(operand, dummy);
        }
      }
    }

    // Clone the ops into the preprocessing func. Note that we could instead
    // clone the entire func and prune ops we don't want to clone, but this is
    // more efficient.
    for (auto [srcRegion, targetRegion] :
         llvm::zip(op->getRegions(), funcOp->getRegions())) {
      for (Block& srcBlock : srcRegion) {
        Block* targetBlock;
        if (&srcBlock == &srcRegion.front()) {
          targetBlock = &targetRegion.front();
        } else {
          targetBlock = builder.createBlock(&targetRegion);
          for (BlockArgument arg : srcBlock.getArguments()) {
            targetBlock->addArgument(arg.getType(), arg.getLoc());
            map.map(arg, targetBlock->getArguments().back());
          }
        }

        for (Operation& childOp : srcBlock) {
          if (analysis.opsToClone.contains(&childOp)) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(targetBlock);
            recursiveCloneOpWithFilter(&childOp, map, analysis.opsToClone,
                                       builder);
          }
        }
      }
    }

    // Insert StoreOps after all cloned EncodeOps
    SmallVector<Operation*> clonedEncodes;
    funcOp.walk([&](PlaintextEncodeOpInterface clonedOp) {
      clonedEncodes.push_back(clonedOp);
    });

    for (Operation* clonedEncode : clonedEncodes) {
      auto siteIdAttr = clonedEncode->getAttrOfType<IntegerAttr>(
          "split_preprocessing_site_id");
      assert(
          siteIdAttr &&
          "Expected split_preprocessing_site_id attribute on cloned encode op");
      int32_t siteId = siteIdAttr.getInt();

      SmallVector<Value> indices =
          getContextualLoopIndices(clonedEncode, funcOp.getOperation());
      builder.setInsertionPointAfter(clonedEncode);
      Value valToStore = clonedEncode->getResult(0);
      Type elemStorageTy = valToStore.getType();

      if (!isAllowedPlaintextType(elemStorageTy)) {
        Location loc = clonedEncode->getLoc();
        funcOp.erase();
        return mlir::emitError(loc)
               << "'lwe.rlwe_encode' op with result type " << elemStorageTy
               << " unsupported in split-preprocessing.";
      }

      // After isAllowedPlaintextType, only a tensor<1x!pt> is allowed here, so
      // extract it and store it.
      if (auto tensorTy = dyn_cast<RankedTensorType>(elemStorageTy)) {
        elemStorageTy = tensorTy.getElementType();
        Value zero =
            arith::ConstantIndexOp::create(builder, clonedEncode->getLoc(), 0);
        valToStore = tensor::ExtractOp::create(builder, clonedEncode->getLoc(),
                                               valToStore, ValueRange{zero});
      }

      preprocessing::StoreOp::create(
          builder, clonedEncode->getLoc(), valToStore, storage, indices,
          builder.getI32IntegerAttr(siteId), TypeAttr::get(elemStorageTy));
    }

    builder.setInsertionPointToEnd(&funcOp.getRegion().back());
    func::ReturnOp::create(builder, funcOp.getLoc(), ValueRange{storage});
    return funcOp;
  }

  FailureOr<FuncOp> createPreprocessedFunction(
      FuncOp op, FuncOp preprocessingFuncOp,
      const PreprocessingAnalysis& analysis) {
    MLIRContext* context = op.getContext();
    OpBuilder builder(context);

    // Add the new preprocessing.storage type as a function argument
    SmallVector<Type> inputTypes(op.getArgumentTypes().begin(),
                                 op.getArgumentTypes().end());
    inputTypes.push_back(preprocessingFuncOp.getResultTypes()[0]);
    auto funcType = FunctionType::get(context, inputTypes, op.getResultTypes());
    auto funcName = op.getName().str() + "__preprocessed";
    auto funcOp = FuncOp::create(op.getLoc(), funcName, funcType);
    funcOp.setVisibility(op.getVisibility());
    funcOp->setAttr(
        kClientPreprocessedFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(op.getName())),
        }));
    funcOp->setAttr(
        kServerEvaluateFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(op.getName())),
        }));
    op->removeAttr(kServerEvaluateFuncAttrName);

    IRMapping map;
    Block* entryBlock = funcOp.addEntryBlock();
    for (auto [idx, arg] : llvm::enumerate(op.getArguments())) {
      map.map(arg, entryBlock->getArgument(idx));
    }
    Value storageArg = entryBlock->getArguments().back();

    builder.setInsertionPointToEnd(entryBlock);
    for (auto& toClone : op.getOps()) {
      Operation* clonedOp = builder.clone(toClone, map);
      for (auto [idx, res] : llvm::enumerate(toClone.getResults())) {
        map.map(res, clonedOp->getResult(idx));
      }
    }

    SmallVector<Operation*> clonedEncodes;
    funcOp.walk([&](PlaintextEncodeOpInterface clonedOp) {
      clonedEncodes.push_back(clonedOp);
    });

    for (Operation* clonedEncode : clonedEncodes) {
      auto siteIdAttr = clonedEncode->getAttrOfType<IntegerAttr>(
          "split_preprocessing_site_id");
      assert(
          siteIdAttr &&
          "Expected split_preprocessing_site_id attribute on cloned encode op");
      int32_t siteId = siteIdAttr.getInt();

      SmallVector<Value> indices =
          getContextualLoopIndices(clonedEncode, funcOp.getOperation());
      builder.setInsertionPointAfter(clonedEncode);
      Type resultTy = clonedEncode->getResult(0).getType();

      if (!isAllowedPlaintextType(resultTy)) {
        Location loc = clonedEncode->getLoc();
        funcOp.erase();
        return mlir::emitError(loc)
               << "'lwe.rlwe_encode' op with result type " << resultTy
               << " unsupported in split-preprocessing.";
      }

      Type loadTy = getElementTypeOrSelf(resultTy);
      auto loadOp = preprocessing::LoadOp::create(
          builder, clonedEncode->getLoc(), loadTy, storageArg, indices,
          builder.getI32IntegerAttr(siteId), TypeAttr::get(loadTy));
      Value loadedVal = loadOp.getResult();

      // After isAllowedPlaintextType, only a tensor<1x!pt> is supported, so
      // reconstruct it from the loaded value.
      if (isa<RankedTensorType>(resultTy)) {
        loadedVal = tensor::FromElementsOp::create(
            builder, clonedEncode->getLoc(), resultTy, ValueRange{loadedVal});
      }

      clonedEncode->getResult(0).replaceAllUsesWith(loadedVal);
      clonedEncode->erase();
    }

    return funcOp;
  }

  PreprocessingAnalysis analyzePreprocessing(FuncOp funcOp) {
    PreprocessingAnalysis analysis;

    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<PlaintextEncodeOpInterface>(op)) {
        analysis.encodeOps.push_back(op);
      }
    });
    if (analysis.encodeOps.empty()) {
      return analysis;
    }

    analysis.opsToClone =
        computeDependenciesToClone(analysis.encodeOps, funcOp);

    // Gather any required block arguments for inputs. These are all expected to
    // be args of the func.func, but we filter out Ciphertext types.
    for (auto* op : analysis.opsToClone) {
      for (auto arg : op->getOperands()) {
        auto argOp = arg.getDefiningOp();
        if (analysis.opsToClone.count(argOp) ||
            isa<lwe::LWECiphertextType>(getElementTypeOrSelf(arg.getType()))) {
          continue;
        }

        if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
          Operation* parentOp = blockArg.getOwner()->getParentOp();
          if (analysis.opsToClone.contains(parentOp)) {
            continue;
          }

          if (parentOp != funcOp.getOperation()) {
            parentOp->emitWarning()
                << "split-preprocessing identified a block argument input "
                << " that should be cloned into the preprocessing function, "
                   "but it was not a block argument of the parent func "
                << funcOp.getName() << ". The input was " << blockArg;
          }
        }

        analysis.inputs.insert(arg);
      }
    }

    return analysis;
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
