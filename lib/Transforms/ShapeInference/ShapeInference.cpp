#include "lib/Transforms/ShapeInference/ShapeInference.h"

#include <memory>

#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "shape-inference"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SHAPEINFERENCE
#include "lib/Transforms/ShapeInference/ShapeInference.h.inc"

struct ConvertFuncArguments : public OpRewritePattern<func::FuncOp> {
  ConvertFuncArguments(MLIRContext *context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    // fetch the argument attrs (if there are none, we don't need to continue)
    auto optionalAttrs = funcOp.getArgAttrs();
    if (!optionalAttrs) return failure();
    auto attrs = optionalAttrs->getValue();

    bool changed = false;

    for (auto arg : funcOp.getArguments()) {
      auto rankedType = dyn_cast<RankedTensorType>(arg.getType());
      if (!rankedType || rankedType.hasStaticShape()) continue;

      // Get the dictionary attribute for the argument
      auto dictattr = dyn_cast<DictionaryAttr>(attrs[arg.getArgNumber()]);
      if (!dictattr) continue;

      // Search for the "shape.shape" attribute
      SmallVector<int64_t> shape = {};
      for (auto attr : dictattr) {
        if (attr.getName() == "shape.shape") {
          auto range =
              cast<ArrayAttr>(attr.getValue()).getAsValueRange<IntegerAttr>();
          for (auto x : range) {
            shape.push_back(x.getLimitedValue());
          }
          break;
        }
      }
      if (shape.empty())
        return emitError(
            funcOp->getLoc(),
            "No shape attribute found for argument " +
                std::to_string(arg.getArgNumber()) +
                " despite being a ranked tensor with dynamic shape");

      // Update the type of the actual block argument
      auto newRankedType =
          RankedTensorType::get(shape, rankedType.getElementType());
      arg.setType(newRankedType);

      // Update the type in the "function type"
      auto inputs = funcOp.getFunctionType().getInputs();
      SmallVector<Type> newInputs(inputs.begin(), inputs.end());
      newInputs[arg.getArgNumber()] = newRankedType;
      auto newFunctionType = rewriter.getFunctionType(
          newInputs, funcOp.getFunctionType().getResults());
      funcOp.setFunctionType(newFunctionType);

      bool changed = true;
    }

    // Remove all "shape.shape" arguments from the function
    // (for improved IR readability in future passes)
    clearAttrs(funcOp, "shape.shape");

    // Return failure if no changes were made, to avoid infinite loops
    return success(changed);
  }
};

struct ConvertFuncReturn : public OpRewritePattern<func::FuncOp> {
  ConvertFuncReturn(MLIRContext *context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    // Get Function Signature return types
    auto returnTypes = funcOp.getFunctionType().getResults();

    // Get terminator (func.return) types
    auto terminatorTypes =
        funcOp.getBody().getBlocks().front().getTerminator()->getOperandTypes();

    assert(returnTypes.size() == terminatorTypes.size() &&
           "Expected same number of results in signature and terminator.");

    for (size_t i = 0; i < terminatorTypes.size(); ++i) {
      if (returnTypes[i] == terminatorTypes[i]) continue;
      auto rankedReturnType = dyn_cast<RankedTensorType>(returnTypes[i]);
      if (!rankedReturnType || rankedReturnType.hasStaticShape()) continue;
      auto rankedTerminatorType =
          dyn_cast<RankedTensorType>(terminatorTypes[i]);
      // The first case should never happen,
      // but the second case will happen until the body has been rewritten
      if (!rankedTerminatorType || !rankedTerminatorType.hasStaticShape())
        continue;
      // FIXME: actually update the type!
    }

    return success(changed);
  }
};

// FIXME: How to go about doing the "boring bit" body conversion????
// struct InferReturnType : public RewritePattern {
//   using RewritePattern::RewritePattern;

//   LogicalResult matchAndRewrite(Operation *op,
//                                 PatternRewriter &rewriter) const override {
//     auto inferOp = dyn_cast<InferTypeOpInterface>(op);
//     if (!inferOp) return failure();
//     llvm::dbgs() << "Found inferable op " << op << "\n";

//     SmallVector<Type> inferredReturnTypes;
//     if (failed(inferOp.inferReturnTypes(
//             rewriter.getContext(), op->getLoc(), op->getOperands(),
//             op->getAttrDictionary(), op->getPropertiesStorage(),
//             op->getRegions(), inferredReturnTypes)))
//       return emitError(op->getLoc(), "Failed to infer return types");

//     return failure();
//   }
// };

struct ShapeInference : impl::ShapeInferenceBase<ShapeInference> {
  using ShapeInferenceBase::ShapeInferenceBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertFuncArguments>(context);
    // patterns.add<ConvertFuncReturn>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
