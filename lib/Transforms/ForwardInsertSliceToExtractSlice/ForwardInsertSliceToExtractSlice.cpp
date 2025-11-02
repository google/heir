#include "lib/Transforms/ForwardInsertSliceToExtractSlice/ForwardInsertSliceToExtractSlice.h"

#include <cstdint>
#include <utility>

#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
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
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "forward-insert-slice-to-extract-slice"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FORWARDINSERTSLICETOEXTRACTSLICE
#include "lib/Transforms/ForwardInsertSliceToExtractSlice/ForwardInsertSliceToExtractSlice.h.inc"

FailureOr<OpFoldResult> ForwardSingleInsertSliceToExtractSlice::getValueAtSlice(
    tensor::ExtractSliceOp originalExtractOp,
    TypedValue<RankedTensorType> tensor, PatternRewriter& rewriter) const {
  auto* def = tensor.getDefiningOp();
  if (!def) {
    LLVM_DEBUG(llvm::dbgs() << "No defining op for the extract op\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Considering def for forwarding: " << *def
                          << "\n");

  return llvm::TypeSwitch<Operation&, FailureOr<OpFoldResult>>(*def)
      .Case<tensor::InsertSliceOp>(
          [&](tensor::InsertSliceOp insertOp) -> FailureOr<OpFoldResult> {
            // Check if the slice is the same as the insert slice, then return
            // that.
            auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
            if (insertOp &&
                insertOp.getSource().getType() == originalExtractOp.getType() &&
                insertOp.isSameAs(originalExtractOp, isSame))
              return getAsOpFoldResult(insertOp.getSource());

            // If it's exactly disjoint, return the insertion's destination and
            // continue.
            auto insertSliceType =
                cast<RankedTensorType>(insertOp.getSource().getType());
            auto insertDestType =
                cast<RankedTensorType>(insertOp.getDest().getType());
            auto insertRel = getSliceInsertionRelation(
                insertSliceType, insertDestType,
                SmallVector<int64_t>(insertOp.getStaticOffsets()),
                SmallVector<int64_t>(insertOp.getStaticSizes()),
                SmallVector<int64_t>(insertOp.getStaticStrides()));
            if (failed(insertRel)) return failure();

            auto extractResultType =
                cast<RankedTensorType>(originalExtractOp.getResult().getType());
            auto extractSourceType =
                cast<RankedTensorType>(originalExtractOp.getSource().getType());
            auto extractRel = getSliceInsertionRelation(
                extractResultType, extractSourceType,
                SmallVector<int64_t>(originalExtractOp.getStaticOffsets()),
                SmallVector<int64_t>(originalExtractOp.getStaticSizes()),
                SmallVector<int64_t>(originalExtractOp.getStaticStrides()));
            if (failed(extractRel)) return failure();

            insertRel->projectOut(0, insertRel->getNumDomainVars());
            extractRel->projectOut(0, extractRel->getNumDomainVars());
            auto intersection = insertRel->intersect(*extractRel);

            if (intersection.isEmpty()) {
              LLVM_DEBUG(llvm::dbgs()
                         << "insert slice and extract slice op have disjoint "
                            "slices, continuing to defining op\n");
              return getValueAtSlice(originalExtractOp, insertOp.getDest(),
                                     rewriter);
            }

            // Slices are not disjoint and not the same. This is a partial
            // overlap, which is not handled.
            return failure();
          })
      .Case<tensor::EmptyOp>(
          [&](tensor::EmptyOp emptyOp) -> FailureOr<OpFoldResult> {
            // If the slice originated from an empty tensor, then return an
            // empty slice.
            auto sliceType = originalExtractOp.getResultType();
            return getAsOpFoldResult(tensor::EmptyOp::create(
                rewriter, emptyOp.getLoc(), sliceType.getShape(),
                sliceType.getElementType()));
          })
      .Default([&](Operation&) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported defining operation, cannot "
                      "traverse backwards to check for forwardable values\n");
        return failure();
      });
}

LogicalResult ForwardSingleInsertSliceToExtractSlice::matchAndRewrite(
    tensor::ExtractSliceOp extractOp, PatternRewriter& rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Considering extractOp for replacement: "
                          << extractOp << "\n");

  auto result = getValueAtSlice(extractOp, extractOp.getSource(), rewriter);
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

struct ForwardInsertSliceToExtractSlice
    : impl::ForwardInsertSliceToExtractSliceBase<
          ForwardInsertSliceToExtractSlice> {
  using ForwardInsertSliceToExtractSliceBase::
      ForwardInsertSliceToExtractSliceBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ForwardSingleInsertSliceToExtractSlice>(context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
