#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h"

#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "forward-insert-to-extract"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FORWARDINSERTTOEXTRACT
#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h.inc"

FailureOr<OpFoldResult> ForwardSingleInsertToExtract::getValueAtIndex(
    TypedValue<RankedTensorType> tensor,
    SmallVector<OpFoldResult> indices) const {
  auto* def = tensor.getDefiningOp();
  if (!def) {
    LLVM_DEBUG(llvm::dbgs() << "No defining op for the extract op\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Considering def for forwarding: " << *def
                          << "\n");

  return llvm::TypeSwitch<Operation&, FailureOr<OpFoldResult>>(*def)
      .Case<tensor::InsertOp>(
          [&](tensor::InsertOp insertOp) -> FailureOr<OpFoldResult> {
            // Check if indices match. If not, continue to defining op of
            // the insertion.
            auto insertIndices = getAsOpFoldResult(insertOp.getIndices());
            if (isEqualConstantIntOrValueArray(insertIndices, indices)) {
              // Found a match, so retrieve the value inserted.
              LLVM_DEBUG(llvm::dbgs()
                         << "insert and op matches extracted indices: "
                         << insertOp << "\n");
              return getAsOpFoldResult(insertOp.getScalar());
            }

            LLVM_DEBUG(llvm::dbgs()
                       << "insert and extract op do not have matching "
                          "indices, continuing to defining op\n");
            return getValueAtIndex(insertOp.getDest(), indices);
          })
      .Case<arith::ConstantOp>([&](auto constantOp) -> FailureOr<OpFoldResult> {
        // If this is a splat elements attribute, simply return the value.
        if (auto splatTensor =
                llvm::dyn_cast<SplatElementsAttr>(constantOp.getValue()))
          return OpFoldResult(splatTensor.template getSplatValue<Attribute>());
        // Collect the constant indices into the tensor.
        auto maybeConstIndices = getConstantIntValues(indices);
        if (!maybeConstIndices.has_value()) return failure();
        auto constIndices = llvm::map_to_vector(
            maybeConstIndices.value(),
            [](int64_t i) { return static_cast<uint64_t>(i); });
        // If this is an elements attribute, query the value at the given
        // indices.
        auto elementsAttr = llvm::dyn_cast<ElementsAttr>(constantOp.getValue());
        if (elementsAttr && elementsAttr.isValidIndex(constIndices))
          return OpFoldResult(
              elementsAttr.template getValues<Attribute>()[constIndices]);

        return failure();
      })
      .Case<tensor::FromElementsOp>([&](tensor::FromElementsOp fromElementsOp)
                                        -> FailureOr<OpFoldResult> {
        // Get the element at the flattened index of the tensor.
        auto flatIndex = getFlattenedIndex(fromElementsOp.getType(), indices);
        if (failed(flatIndex)) return failure();
        return OpFoldResult(fromElementsOp.getElements()[flatIndex.value()]);
      })
      .Default([&](Operation&) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported defining operation, cannot "
                      "traverse backwards to check for forwardable values\n");
        return failure();
      });
}

LogicalResult ForwardSingleInsertToExtract::matchAndRewrite(
    tensor::ExtractOp extractOp, PatternRewriter& rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Considering extractOp for replacement: "
                          << extractOp << "\n");

  auto result = getValueAtIndex(extractOp.getTensor(),
                                getAsOpFoldResult(extractOp.getIndices()));
  if (failed(result)) {
    LLVM_DEBUG(llvm::dbgs() << "no forwardable values found: \n");
    return failure();
  }
  OpFoldResult forwardedResult = result.value();
  if (auto forwardedValue = dyn_cast<Value>(forwardedResult)) {
    rewriter.replaceAllUsesWith(extractOp, forwardedValue);
  } else {
    auto forwardedAttr = cast<TypedAttr>(cast<Attribute>(forwardedResult));
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        extractOp, forwardedAttr.getType(), forwardedAttr);
  }
  return success();
}

struct ForwardInsertToExtract
    : impl::ForwardInsertToExtractBase<ForwardInsertToExtract> {
  using ForwardInsertToExtractBase::ForwardInsertToExtractBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    DominanceInfo dom(getOperation());
    patterns.add<ForwardSingleInsertToExtract>(context, dom);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
