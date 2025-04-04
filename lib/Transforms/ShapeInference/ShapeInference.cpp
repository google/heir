#include "lib/Transforms/ShapeInference/ShapeInference.h"

#include <memory>

#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "shape-inference"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SHAPEINFERENCE
#include "lib/Transforms/ShapeInference/ShapeInference.h.inc"

struct InferShape : public OpRewritePattern<func::FuncOp> {
  InferShape(MLIRContext *context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    /////////////////////////////////////////////////////////////////////
    //// Step 1: Check for shape-annotated dynamic-sized tensor args ////
    /////////////////////////////////////////////////////////////////////

    // find all non-static-shape ranked tensor types in the function's arguments
    std::vector<std::tuple<int, RankedTensorType>> dynTensorArgs = {};
    for (auto [idx, type] :
         llvm::enumerate(funcOp.getFunctionType().getInputs())) {
      auto rankedType = dyn_cast<RankedTensorType>(type);
      if (!rankedType || rankedType.hasStaticShape()) continue;
      dynTensorArgs.emplace_back(idx, rankedType);
    }
    // no non-static-shape tensor types -> exit
    if (dynTensorArgs.empty()) return failure();

    // Check if there are even argument attributes at all.
    // If not, there is nothing to infer from -> exit
    auto optionalAttrs = funcOp.getArgAttrs();
    if (!optionalAttrs) return failure();
    auto attrs = optionalAttrs->getValue();

    // Update types with annotations (where it exists)
    bool changed = false;
    for (auto &[idx, type] : dynTensorArgs) {
      // Check if the current argument has any attributes.
      auto dictattr = dyn_cast<DictionaryAttr>(attrs[idx]);
      if (!dictattr) continue;  // nothing to infer from

      // Get the shape attribute if it exists
      auto *shapeAttr = llvm::find_if(dictattr, [&](NamedAttribute attr) {
        return attr.getName() == "shape.shape";
      });
      if (!shapeAttr) continue;  // nothing to infer from

      // Extract the actual shape from the shape annotation attribute
      auto arrayAttr = dyn_cast<ArrayAttr>(shapeAttr->getValue());
      if (!arrayAttr)
        return emitError(funcOp.getLoc(),
                         "Invalid shape attribute (expected ArrayAttr)");

      if (llvm::any_of(arrayAttr,
                       [](Attribute attr) { return !isa<IntegerAttr>(attr); }))
        return emitError(
            funcOp.getLoc(),
            "Invalid shape attribute (expected ArrayAttr of only IntegerAttr)");

      SmallVector<int64_t> shape = llvm::to_vector(
          llvm::map_range(arrayAttr.getAsRange<IntegerAttr>(),
                          [](IntegerAttr i) { return i.getInt(); }));

      if (shape.size() != type.getRank())
        return emitError(
            funcOp.getLoc(),
            llvm::formatv("Shape annotation has wrong number of elements "
                          "(expected {0}, got {1}).",
                          type.getRank(), shape.size()));

      // save the updated argument type
      type = RankedTensorType::get(shape, type.getElementType());
      changed = true;
    }
    // no new information found to infer from -> exit
    if (!changed) return failure();

    rewriter.startOpModification(funcOp);

    /////////////////////////////////////////////////////////////////////
    ////  Step 2: Update Function Type/Signature based on Attributes ////
    /////////////////////////////////////////////////////////////////////
    // We need to update the function type to reflect the new types,
    // update the type of the "real" arguments (block arguments),
    // and we also remove the attribute argument from the signature
    // (primarily for better readability of the IR after shape inference).
    //
    // We could delay updating the function type until the end,
    // when we also know the inferred return type(s),
    // but that would leave the func op in a less consistent state longer
    auto newInputTypes = llvm::to_vector(funcOp.getFunctionType().getInputs());
    for (auto [idx, type] : dynTensorArgs) {
      funcOp.getArgument(idx).setType(type);
      funcOp.removeArgAttr(idx, "shape.shape");
      newInputTypes[idx] = type;
    }
    auto newFunctionType = rewriter.getFunctionType(
        newInputTypes, funcOp.getFunctionType().getResults());
    funcOp.setFunctionType(newFunctionType);

    /////////////////////////////////////////////////////////////////////
    //// Step 3: Walk the body, inferring and updating new types     ////
    /////////////////////////////////////////////////////////////////////

    auto walkResult = funcOp.getBody().walk([&](Operation *op) {
      // We focus on operations with return type inference,
      // as these will likely complain if their operands' types change
      // but their return type hasn't been updated.
      // Technically, other ops could also cause issues,
      // For example, an op could allow only dynamically shaped tensors,
      // which means it would fail to verify after shape inference,
      // but failing on those is probably the right thing to do for the compiler
      if (auto inferOp = dyn_cast<InferTypeOpInterface>(op)) {
        SmallVector<Type> inferredReturnTypes;
        if (failed(inferOp.inferReturnTypes(
                rewriter.getContext(), op->getLoc(), op->getOperands(),
                op->getAttrDictionary(), op->getPropertiesStorage(),
                op->getRegions(), inferredReturnTypes))) {
          emitError(op->getLoc(), "Failed to infer return types.");
          return WalkResult::interrupt();
        }

        assert(
            inferredReturnTypes.size() == op->getNumResults() &&
            "Number of inferred return types must match the number of results");

        for (size_t i = 0; i < inferredReturnTypes.size(); ++i) {
          if (inferredReturnTypes[i] == op->getResult(i).getType()) continue;
          op->getResult(i).setType(inferredReturnTypes[i]);
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      rewriter.cancelOpModification(funcOp);
      return failure();
    }

    /////////////////////////////////////////////////////////////////////
    //// Step 4: Match the return type(s) in the signature           ////
    /////////////////////////////////////////////////////////////////////

    // Get Function Signature return types
    auto returnTypes = funcOp.getFunctionType().getResults();

    // Get terminator (func.return) types
    auto *terminator = funcOp.getBody().getBlocks().front().getTerminator();
    assert(terminator && "missing required terminator in func op.");
    auto terminatorTypes = terminator->getOperandTypes();

    assert(returnTypes.size() == terminatorTypes.size() &&
           "Expected same number of results in signature and terminator.");

    funcOp.setFunctionType(rewriter.getFunctionType(
        funcOp.getFunctionType().getInputs(), terminatorTypes));

    // Signal that we have successfully modified the op
    rewriter.finalizeOpModification(funcOp);
    return success();
  }
};

struct ShapeInference : impl::ShapeInferenceBase<ShapeInference> {
  using ShapeInferenceBase::ShapeInferenceBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // FIXME: This probably shouldn't even be a pattern anymore...
    patterns.add<InferShape>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
