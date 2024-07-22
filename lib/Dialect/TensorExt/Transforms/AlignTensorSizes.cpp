#include "lib/Dialect/TensorExt/Transforms/AlignTensorSizes.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_NAME "align-tensor-sizes"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_ALIGNTENSORSIZES
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

class SecretTensorTypeConverter : public TypeConverter {
 public:
  SecretTensorTypeConverter(int size) {
    addConversion([](Type type) { return type; });

    addConversion([&](RankedTensorType type) -> std::optional<Type> {
      auto ctx = type.getContext();
      if (dyn_cast_or_null<SIMDPackingAttr>(type.getEncoding())) {
        // This type already has a SIMD packing attribute.
        return type;
      }

      auto dimension = type.getShape()[0];
      if (dimension != size_) {
        SmallVector<int64_t> newShape = {this->size_};
        if (dimension > size_) {
          // Split the tensor into a multi-dimensional tensor.
          newShape = {static_cast<int64_t>(llvm::divideCeil(dimension, size_)),
                      size_};
        }
        auto padding = DenseI64ArrayAttr::get(
            ctx,
            llvm::ArrayRef<int64_t>(llvm::PowerOf2Ceil(dimension) - dimension));

        return RankedTensorType::get(
            newShape, type.getElementType(),
            SIMDPackingAttr::get(
                ctx, /*in=*/DenseI64ArrayAttr::get(ctx, type.getShape()),
                padding,
                /*out=*/
                DenseI64ArrayAttr::get(ctx, newShape), /*padding_value=*/0));
      }
      return type;
    });

    addConversion([&](secret::SecretType secretType) -> Type {
      auto convertedTensorType = this->convertType(secretType.getValueType());
      return secret::SecretType::get(convertedTensorType);
    });

    size_ = size;
  }
  int size_;
};

struct ConvertTensorExtractOp : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertTensorExtractOp(MLIRContext *context)
      : OpConversionPattern<tensor::ExtractOp>(context, 2) {}

  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto newTensorType = cast<RankedTensorType>(adaptor.getTensor().getType());

    auto size = newTensorType.getShape()[1];
    auto index =
        cast<arith::ConstantIndexOp>(op.getIndices()[0].getDefiningOp())
            .value();
    SmallVector<Value> newIndices(
        {rewriter.create<arith::ConstantIndexOp>(op.getLoc(), index / size),
         rewriter.create<arith::ConstantIndexOp>(op.getLoc(), index % size)});

    auto newOp = rewriter.create<tensor::ExtractOp>(
        op->getLoc(), newTensorType.getElementType(), adaptor.getTensor(),
        newIndices);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertTensorInsertOp : public OpConversionPattern<tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertTensorInsertOp(MLIRContext *context)
      : OpConversionPattern<tensor::InsertOp>(context, 2) {}

  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto newTensorType = cast<RankedTensorType>(adaptor.getDest().getType());

    auto size = newTensorType.getShape()[1];
    auto index =
        cast<arith::ConstantIndexOp>(op.getIndices()[0].getDefiningOp())
            .value();
    SmallVector<Value> newIndices(
        {rewriter.create<arith::ConstantIndexOp>(op.getLoc(), index / size),
         rewriter.create<arith::ConstantIndexOp>(op.getLoc(), index % size)});

    auto newOp = rewriter.create<tensor::InsertOp>(
        op->getLoc(), adaptor.getScalar(), adaptor.getDest(), newIndices);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Promote tensor types to a tensor with fixed final dimension and a
// SIMDPackingAttr describing the transformation.
struct AlignTensorSizes : impl::AlignTensorSizesBase<AlignTensorSizes> {
  using AlignTensorSizesBase::AlignTensorSizesBase;

  void runOnOperation() override {
    // Pass currently requires that all tensors are 1-D. A smarter pass should
    // run before this one to customize encoding large dimensional plaintexts
    // and lowering of higher-level ops.
    auto result = getOperation()->walk([](Operation *op) {
      auto multidimensional =
          llvm::any_of(op->getOperandTypes(), [](Type type) {
            if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
              return tensorType.getRank() != 1;
            }
            return false;
          });
      multidimensional |= llvm::any_of(op->getResultTypes(), [](Type type) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
          return tensorType.getRank() != 1;
        }
        return false;
      });
      return multidimensional ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = &getContext();
    SecretTensorTypeConverter typeConverter(size);

    RewritePatternSet patterns(context);
    patterns.add<ConvertAny, ConvertTensorExtractOp, ConvertTensorInsertOp>(
        typeConverter, context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    addStructuralConversionPatterns(typeConverter, patterns, target);
    // Note: addStructuralConversionPatterns includes adding a legality using
    // markUnknownOpDynamicallyLegal for BranchOpInterface, so we override it
    // here. This is a bit hacky.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<tensor::ExtractOp>([&](tensor::ExtractOp op) {
      return typeConverter.isLegal(op) &&
             op.getIndices().size() == op.getTensor().getType().getRank();
    });
    target.addDynamicallyLegalOp<tensor::InsertOp>([&](tensor::InsertOp op) {
      return typeConverter.isLegal(op) &&
             op.getIndices().size() == op.getDest().getType().getRank();
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
