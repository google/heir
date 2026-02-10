#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"

#include <cassert>
#include <memory>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "split-preprocessing"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SPLITPREPROCESSING
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h.inc"

using func::FuncOp;

namespace {

// The following slice analysis code is copied verbatim from
// https://github.com/llvm/circt/blob/main/lib/Dialect/SV/Transforms/SVExtractTestCode.cpp
// except where noted.

// Reimplemented from SliceAnalysis to use a worklist rather than
// recursion and non-insert ordered set.  Implement this as a DFS and not a BFS
// so that the order is stable across changes to intermediary operations.  (It
// is then necessary to use the _operands_ as a worklist and not the
// _operations_.)
static void getBackwardSliceSimple(
    Operation* rootOp, SetVector<Operation*>& backwardSlice,
    llvm::function_ref<bool(Operation*)> filter) {
  SmallVector<Value> worklist(rootOp->getOperands());

  while (!worklist.empty()) {
    Value operand = worklist.pop_back_val();
    Operation* definingOp = operand.getDefiningOp();

    if (!definingOp ||
        definingOp->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      continue;

    // Evaluate whether we should keep this def.
    // This is useful in particular to implement scoping; i.e. return the
    // transitive backwardSlice in the current scope.
    if (filter && !filter(definingOp)) continue;

    if (definingOp) {
      if (!backwardSlice.contains(definingOp))
        for (auto newOperand : llvm::reverse(definingOp->getOperands()))
          worklist.push_back(newOperand);
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Block* block = blockArg.getOwner();
      Operation* parentOp = block->getParentOp();
      // Determine whether we want to recurse backward into the other
      // blocks of parentOp, which are not technically backward unless they
      // flow into us. For now, just bail.
      assert(parentOp->getNumRegions() == 1 &&
             parentOp->getRegion(0).getBlocks().size() == 1);
      if (!backwardSlice.contains(parentOp))
        for (auto newOperand : llvm::reverse(parentOp->getOperands()))
          worklist.push_back(newOperand);
    } else {
      llvm_unreachable("No definingOp and not a block argument.");
    }

    backwardSlice.insert(definingOp);
  }
}

// Compute the ops defining the blocks a set of ops are in.
static void blockSlice(SetVector<Operation*>& ops,
                       SetVector<Operation*>& blocks) {
  for (auto op : ops) {
    while (!isa<func::FuncOp>(op->getParentOp())) {
      op = op->getParentOp();
      blocks.insert(op);
    }
    // Differing from SV implementation: Add ops within regions to the worklist
    // to ensure that the dataflow to operands used within regions is captured.
    op->walk([&](Operation* op) { blocks.insert(op); });
  }
}

static void computeSlice(SetVector<Operation*>& roots,
                         SetVector<Operation*>& results,
                         llvm::function_ref<bool(Operation*)> filter) {
  for (auto* op : roots) getBackwardSliceSimple(op, results, filter);
}

// Return a backward slice started from `roots` until dataflow reaches to an
// operations for which `filter` returns false.
static SetVector<Operation*> getBackwardSlice(
    SetVector<Operation*>& roots, llvm::function_ref<bool(Operation*)> filter) {
  SetVector<Operation*> results;
  computeSlice(roots, results, filter);

  // Get Blocks
  SetVector<Operation*> blocks;
  blockSlice(roots, blocks);
  blockSlice(results, blocks);

  // Make sure dataflow to block args (if conds, etc) and ops within regions are
  // included
  computeSlice(blocks, results, filter);
  // Differing from SV implementation: don't insert the operations within the
  // regions (since their parent op is already in the set).
  results.insert(roots.begin(), roots.end());
  return results;
}

struct PreprocessingAnalysis {
  SetVector<Operation*> encodeOps;
  // Use a vector to preserve insertion order for stable function signatures.
  SetVector<Value> inputs;
  SetVector<Operation*> opsToClone;
};

struct SplitPreprocessingPass
    : impl::SplitPreprocessingBase<SplitPreprocessingPass> {
  using SplitPreprocessingBase::SplitPreprocessingBase;

  void runOnOperation() override {
    Operation* root = getOperation();
    root->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
      if (op.isDeclaration() || isClientHelper(op)) {
        return;
      }
      convertFunc(op);
    });
  }

  void convertFunc(FuncOp funcOp) {
    PreprocessingAnalysis analysis = analyzePreprocessing(funcOp);
    if (analysis.encodeOps.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "No encode ops found in function " << funcOp.getName()
                     << "\n";
      });
      return;
    } else if (analysis.encodeOps.size() > 512) {
      // TODO(#2603): Grouping plaintexts will reduce the number of function
      // return types.
      LLVM_DEBUG({
        llvm::dbgs() << "Too many encode ops (" << analysis.encodeOps.size()
                     << ") found in function " << funcOp.getName() << "\n";
      });
      return;
    }

    FuncOp preprocessingFuncOp = createPreprocessingFunction(funcOp, analysis);
    FuncOp preprocessedFuncOp = createPreprocessedFunction(funcOp, analysis);

    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    moduleOp.insert(funcOp.getOperation(), preprocessingFuncOp);
    moduleOp.insert(funcOp.getOperation(), preprocessedFuncOp);

    updateOriginalFunc(funcOp, preprocessingFuncOp, preprocessedFuncOp,
                       analysis);
  }

  void updateOriginalFunc(FuncOp funcOp, FuncOp preprocessingFuncOp,
                          FuncOp preprocessedFuncOp,
                          const PreprocessingAnalysis& analysis) {
    // Add calls to the preprocessing and preprocessed functions.
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    auto preprocessingCall =
        func::CallOp::create(builder, funcOp.getLoc(), preprocessingFuncOp,
                             llvm::to_vector(analysis.inputs));

    SmallVector<Value> preprocessedArgs;
    for (auto arg : funcOp.getArguments()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(arg.getType()))) {
        preprocessedArgs.push_back(arg);
      }
    }
    preprocessedArgs.append(preprocessingCall.getResults().begin(),
                            preprocessingCall.getResults().end());
    auto preprocessedCall = func::CallOp::create(
        builder, funcOp.getLoc(), preprocessedFuncOp, preprocessedArgs);

    Block* originalEntry = &funcOp.getBody().front();
    Operation* originalTerminator = originalEntry->getTerminator();
    originalTerminator->setOperands(preprocessedCall.getResults());

    // At this point all operations should have been moved and we can remove all
    // ops but the calls.
    DenseSet<Operation*> opsToKeep = {preprocessingCall, preprocessedCall,
                                      originalTerminator};
    for (Operation& op : llvm::make_early_inc_range(
             llvm::reverse(originalEntry->getOperations()))) {
      if (!opsToKeep.contains(&op)) op.erase();
    }
  }

  FuncOp createPreprocessingFunction(FuncOp op,
                                     const PreprocessingAnalysis& analysis) {
    MLIRContext* context = op.getContext();
    OpBuilder builder(context);

    SmallVector<Type> newInputs;
    for (const auto& input : analysis.inputs) {
      newInputs.push_back(input.getType());
    }

    SmallVector<Type> newResults;
    for (const auto& output : analysis.encodeOps) {
      newResults.push_back(output->getResult(0).getType());
    }

    auto funcType = FunctionType::get(context, newInputs, newResults);
    auto funcName = op.getName().str() + "__preprocessing";
    auto funcOp = FuncOp::create(op.getLoc(), funcName, funcType);
    funcOp.setVisibility(op.getVisibility());
    // Add a special attribute to the new function so that
    // later passes can reference the original function.
    funcOp->setAttr(
        kClientPackFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(op.getName())),
        }));

    IRMapping map;
    Block* entryBlock = funcOp.addEntryBlock();
    for (auto [idx, input] : llvm::enumerate(analysis.inputs)) {
      map.map(input, entryBlock->getArgument(idx));
    }

    // Insert the preprocessing ops into the new function.
    builder.setInsertionPointToEnd(entryBlock);
    for (auto& op : op.getOps()) {
      if (!analysis.opsToClone.contains(&op)) {
        continue;
      }
      builder.clone(op, map);
    }
    // TODO(#2603): Find a smart way to group together encoded plaintexts.
    SmallVector<Value> results;
    for (const auto& encodeOp : analysis.encodeOps) {
      auto result = map.lookup(encodeOp->getResults()[0]);
      results.push_back(result);
    }
    func::ReturnOp::create(builder, funcOp.getLoc(), results);
    return funcOp;
  }

  FuncOp createPreprocessedFunction(FuncOp op,
                                    const PreprocessingAnalysis& analysis) {
    MLIRContext* context = op.getContext();
    OpBuilder builder(context);

    SmallVector<Type> inputTypes;
    for (auto argType : op.getArgumentTypes()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(argType))) {
        inputTypes.push_back(argType);
      }
    }
    for (auto encodeOp : analysis.encodeOps) {
      inputTypes.push_back(encodeOp->getResult(0).getType());
    }
    auto funcType = FunctionType::get(context, inputTypes, op.getResultTypes());
    auto funcName = op.getName().str() + "__preprocessed";
    auto funcOp = FuncOp::create(op.getLoc(), funcName, funcType);
    funcOp.setVisibility(op.getVisibility());
    // Add a special attribute to the new function so that
    // later passes can reference the original function.
    funcOp->setAttr(
        kClientPreprocessedFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(op.getName())),
        }));

    // Build the main function body in the preprocessed function.
    IRMapping map;
    Block* entryBlock = funcOp.addEntryBlock();
    int index = 0;
    for (auto arg : op.getArguments()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(arg.getType()))) {
        map.map(arg, entryBlock->getArgument(index++));
      }
    }
    // Map each of the encode ops to a block argument.
    for (auto encodeOp : analysis.encodeOps) {
      auto plaintextArg = entryBlock->getArgument(index++);
      map.map(encodeOp->getResult(0), plaintextArg);
    }

    builder.setInsertionPointToEnd(entryBlock);

    for (auto& toClone : op.getOps()) {
      if (analysis.opsToClone.contains(&toClone) &&
          !toClone.hasTrait<mlir::OpTrait::ConstantLike>()) {
        // Perhaps this is brittle? The for loop intends to copy all ops that
        // are not exclusively used for pre-processing. Otherwise we would need
        // to iterate over all uses of the ops.
        continue;
      }
      builder.clone(toClone, map);
    }
    return funcOp;
  }

  PreprocessingAnalysis analyzePreprocessing(FuncOp funcOp) {
    PreprocessingAnalysis analysis;

    for (auto encodeOp : funcOp.getBody().getOps<lwe::RLWEEncodeOp>()) {
      analysis.encodeOps.insert(encodeOp);
    }
    if (analysis.encodeOps.empty()) {
      return analysis;
    }

    analysis.opsToClone =
        getBackwardSlice(analysis.encodeOps, [&](Operation* op) {
          return op->getParentRegion() == &funcOp.getRegion();
        });

    // Gather any required block arguments for inputs.
    for (auto* op : analysis.opsToClone) {
      for (auto arg : op->getOperands()) {
        auto argOp = arg.getDefiningOp();  // may be null
        if (!analysis.opsToClone.count(argOp)) analysis.inputs.insert(arg);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Adding inputs for preprocessing:\n";
      for (auto input : analysis.inputs) {
        llvm::dbgs() << "\t - " << input << "\n";
      }
    });

    return analysis;
  }
};

}  // namespace

std::unique_ptr<Pass> createSplitPreprocessingPass() {
  return std::make_unique<SplitPreprocessingPass>();
}

}  // namespace heir
}  // namespace mlir
