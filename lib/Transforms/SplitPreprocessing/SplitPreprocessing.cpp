#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "split-preprocessing"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SPLITPREPROCESSING
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h.inc"

using func::FuncOp;

namespace {

struct PreprocessingAnalysis {
  SetVector<Operation*> preprocessingOps;
  // Use a vector of pairs to preserve insertion order for stable function
  // signatures.
  SmallVector<std::pair<Value, SmallVector<lwe::RLWEEncodeOp>>>
      cleartextToEncodeOps;
  SmallVector<Type> plaintextTypes;

  bool empty() const { return cleartextToEncodeOps.empty(); }
};

struct SplitPreprocessingPass
    : impl::SplitPreprocessingBase<SplitPreprocessingPass> {
  using SplitPreprocessingBase::SplitPreprocessingBase;

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    if (funcOp.isDeclaration() || isClientHelper(funcOp)) {
      return;
    }

    PreprocessingAnalysis analysis = analyzePreprocessing(funcOp);
    if (analysis.empty()) {
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
    OpBuilder builder(funcOp.getContext());
    // Add call operations in the main function body to the preprocessing
    // and preprocessed functions.
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    SmallVector<Value> callPreprocessingArgs;
    for (const auto& [arg, _] : analysis.cleartextToEncodeOps) {
      if (isa<BlockArgument>(arg)) {
        callPreprocessingArgs.push_back(arg);
      }
    }
    auto callPreprocessingOp = func::CallOp::create(
        builder, funcOp.getLoc(), preprocessingFuncOp, callPreprocessingArgs);

    SmallVector<Value> callPreprocessedArgs;
    for (auto arg : funcOp.getArguments()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(arg.getType()))) {
        callPreprocessedArgs.push_back(arg);
      }
    }
    callPreprocessedArgs.append(callPreprocessingOp.getResults().begin(),
                                callPreprocessingOp.getResults().end());
    auto callPreprocessedOp = func::CallOp::create(
        builder, funcOp.getLoc(), preprocessedFuncOp, callPreprocessedArgs);

    Block* originalEntry = &funcOp.getBody().front();
    Operation* originalTerminator = originalEntry->getTerminator();
    originalTerminator->setOperands(callPreprocessedOp.getResults());

    // At this point all operations should have been moved and we can remove all
    // ops but the calls.
    DenseSet<Operation*> opsToKeep = {callPreprocessingOp, callPreprocessedOp,
                                      originalTerminator};
    for (Operation& op : llvm::make_early_inc_range(
             llvm::reverse(originalEntry->getOperations()))) {
      if (!opsToKeep.contains(&op)) op.erase();
    }
  }

  FuncOp createPreprocessingFunction(FuncOp funcOp,
                                     const PreprocessingAnalysis& analysis) {
    MLIRContext* context = funcOp.getContext();
    OpBuilder builder(context);

    SmallVector<Type> cleartextArgs;
    for (const auto& [arg, _] : analysis.cleartextToEncodeOps) {
      if (isa<BlockArgument>(arg)) {
        cleartextArgs.push_back(arg.getType());
      }
    }
    auto preprocessingFuncType =
        FunctionType::get(context, cleartextArgs, analysis.plaintextTypes);
    auto preprocessingFuncName = funcOp.getName().str() + "__preprocessing";
    auto preprocessingFuncOp = FuncOp::create(
        funcOp.getLoc(), preprocessingFuncName, preprocessingFuncType);
    preprocessingFuncOp.setVisibility(funcOp.getVisibility());
    // Add a client.pack_func attribute to the new function so that
    // later passes can reference the original function.
    preprocessingFuncOp->setAttr(
        kClientPackFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(funcOp.getName())),
        }));

    IRMapping preprocessingMapping;
    Block* preprocessingEntryBlock = preprocessingFuncOp.addEntryBlock();
    int index = 0;
    for (const auto& [arg, _] : analysis.cleartextToEncodeOps) {
      if (isa<BlockArgument>(arg)) {
        preprocessingMapping.map(arg,
                                 preprocessingEntryBlock->getArgument(index++));
      }
    }

    // Insert the preprocessing ops into the new function.
    builder.setInsertionPointToEnd(preprocessingEntryBlock);
    for (auto op : analysis.preprocessingOps) {
      builder.clone(*op, preprocessingMapping);
    }
    // Make tensor.from_elements ops to collect the plaintext results;
    SmallVector<Value> plaintextResults;
    for (const auto& [arg, ops] : analysis.cleartextToEncodeOps) {
      SmallVector<Value> elements;
      for (auto encodeOp : ops) {
        elements.push_back(preprocessingMapping.lookup(encodeOp.getResult()));
      }
      auto resultOp =
          tensor::FromElementsOp::create(builder, funcOp.getLoc(), elements);
      plaintextResults.push_back(resultOp.getResult());
    }
    func::ReturnOp::create(builder, funcOp.getLoc(), plaintextResults);

    return preprocessingFuncOp;
  }

  FuncOp createPreprocessedFunction(FuncOp funcOp,
                                    const PreprocessingAnalysis& analysis) {
    MLIRContext* context = funcOp.getContext();
    OpBuilder builder(context);

    SmallVector<Type> preprocessedArgTypes;
    for (auto argType : funcOp.getArgumentTypes()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(argType))) {
        preprocessedArgTypes.push_back(argType);
      }
    }
    preprocessedArgTypes.append(analysis.plaintextTypes.begin(),
                                analysis.plaintextTypes.end());
    auto preprocessedFuncType = FunctionType::get(context, preprocessedArgTypes,
                                                  funcOp.getResultTypes());
    auto preprocessedFuncName = funcOp.getName().str() + "__preprocessed";
    auto preprocessedFuncOp = FuncOp::create(
        funcOp.getLoc(), preprocessedFuncName, preprocessedFuncType);
    preprocessedFuncOp.setVisibility(funcOp.getVisibility());
    // Add a client.preprocessed_func attribute to the new function so that
    // later passes can reference the original function.
    preprocessedFuncOp->setAttr(
        kClientPreprocessedFuncAttrName,
        builder.getDictionaryAttr({
            builder.getNamedAttr(kClientHelperFuncName,
                                 builder.getStringAttr(funcOp.getName())),
        }));

    // Build the main function body in the preprocessed function.
    IRMapping preprocessedMapping;
    Block* newEntryBlock = preprocessedFuncOp.addEntryBlock();
    int index = 0;
    for (auto arg : funcOp.getArguments()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(arg.getType()))) {
        preprocessedMapping.map(arg, newEntryBlock->getArgument(index++));
      }
    }
    // Map each of the encode ops to a tensor.extract of the block argument.
    builder.setInsertionPointToEnd(newEntryBlock);
    for (const auto& [_, ops] : analysis.cleartextToEncodeOps) {
      auto newTensorPlaintextArg = newEntryBlock->getArgument(index++);
      for (auto [idx, encodeOp] : llvm::enumerate(ops)) {
        auto idxVal =
            arith::ConstantIndexOp::create(builder, funcOp.getLoc(), idx);
        auto extracted =
            tensor::ExtractOp::create(builder, funcOp.getLoc(),
                                      newTensorPlaintextArg, {idxVal})
                .getResult();
        preprocessedMapping.map(encodeOp->getResults()[0], extracted);
      }
    }
    for (auto& op : funcOp.getBody().getOps()) {
      // Skip ops that are exclusively used for preprocessing.
      if (analysis.preprocessingOps.contains(&op) &&
          (isa<lwe::RLWEEncodeOp>(op) ||
           llvm::all_of(op.getUsers(), [&](Operation* user) {
             return analysis.preprocessingOps.contains(user);
           }))) {
        continue;
      }
      builder.clone(op, preprocessedMapping);
    }

    return preprocessedFuncOp;
  }

  PreprocessingAnalysis analyzePreprocessing(FuncOp funcOp) {
    PreprocessingAnalysis analysis;

    SmallVector<Value> candidateCleartexts;
    for (auto arg : funcOp.getBody().getArguments()) {
      if (!isa<lwe::LWECiphertextType>(getElementTypeOrSelf(arg.getType()))) {
        candidateCleartexts.push_back(arg);
      }
    }
    for (auto constantOp : funcOp.getBody().getOps<arith::ConstantOp>()) {
      // This will also end up including constants used as indices for
      // extractions or rotations, so don't include any index constants.
      if (constantOp.getType().isIndex()) {
        continue;
      }
      candidateCleartexts.push_back(constantOp.getResult());
    }

    for (auto candidate : candidateCleartexts) {
      // Traverse a forward slice to find resulting lwe::RLWEEncodeOps from the
      // cleartexts.
      auto parentRegion = candidate.getParentRegion();
      SetVector<Operation*> forwardSlice;
      ForwardSliceOptions options;
      options.inclusive = true;
      options.filter = [&](Operation* op) {
        if (op->getParentRegion() != parentRegion) {
          // Don't include ops inside regions, since we will clone their
          // parent op.
          return false;
        }
        // Stop when we hit an lwe::RLWEEncodeOp.
        return llvm::none_of(op->getOperands(), [](Value operand) {
          return isa_and_nonnull<lwe::RLWEEncodeOp>(operand.getDefiningOp());
        });
      };
      for (Operation* user : candidate.getUsers()) {
        auto userSlice = getSlice(user, {}, options);
        forwardSlice.insert(userSlice.begin(), userSlice.end());
      }
      if (forwardSlice.empty()) continue;

      // The result is a collection of plaintexts that will be output from the
      // preprocessing function.
      SmallVector<lwe::RLWEEncodeOp> encodeOps;
      for (auto op : forwardSlice) {
        if (auto encodeOp = dyn_cast<lwe::RLWEEncodeOp>(op)) {
          encodeOps.push_back(encodeOp);
        }
      }
      if (!encodeOps.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found cleartext to encode: " << candidate << " with "
                   << encodeOps.size() << " encode ops." << "\n");
        if (!llvm::all_equal(
                llvm::map_range(encodeOps, [](lwe::RLWEEncodeOp encodeOp) {
                  return encodeOp.getType();
                }))) {
          // Unlikely, but possible that the cleartext is used for
          // different plaintext encoding types, so we skip it in this case.
          LLVM_DEBUG(llvm::dbgs() << "Plaintexts types from candidate "
                                  << candidate << " do not match, skipping.\n");

          continue;
        }
        analysis.plaintextTypes.push_back(RankedTensorType::get(
            {static_cast<int64_t>(encodeOps.size())}, encodeOps[0].getType()));
        analysis.cleartextToEncodeOps.push_back(
            {candidate, std::move(encodeOps)});
      }
      analysis.preprocessingOps.insert(forwardSlice.begin(),
                                       forwardSlice.end());
    }
    return analysis;
  }
};

}  // namespace

std::unique_ptr<Pass> createSplitPreprocessingPass() {
  return std::make_unique<SplitPreprocessingPass>();
}

}  // namespace heir
}  // namespace mlir
