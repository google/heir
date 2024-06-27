#include "lib/Dialect/TensorExt/Transforms/AlignTensorSizes.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
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
      if (type.getRank() != 1) {
        emitError(UnknownLoc::get(ctx), "expected 1-D tensor, got") << type;
        return std::nullopt;
      }

      auto dimension = type.getShape()[0];
      if (dimension != size_) {
        SmallVector<int64_t> newShape = {this->size_};
        // TODO(#704): Handle splitting a tensor across multiple tensors.
        if (dimension > size_) {
          emitError(UnknownLoc::get(ctx),
                    "pass only supports tensors with dimension less "
                    "than or equal to ")
              << size_ << " , got " << type;
          return std::nullopt;
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

// Promote tensor types to a tensor with fixed final dimension and a
// SIMDPackingAttr describing the transformation.
struct AlignTensorSizes : impl::AlignTensorSizesBase<AlignTensorSizes> {
  using AlignTensorSizesBase::AlignTensorSizesBase;

  void runOnOperation() override {
    // Open questions: should this only run on secret tensor types? For
    // plaintext matrices, we likely need to pack into a ciphertext if it is
    // cheaper than many MulPlains. (e.g. a dot product with a plaintext weight
    // matrix).
    MLIRContext *context = &getContext();
    SecretTensorTypeConverter typeConverter(size);

    RewritePatternSet patterns(context);
    patterns.add<ConvertAny>(typeConverter, context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    addStructuralConversionPatterns(typeConverter, patterns, target);
    // Note: addStructuralConversionPatterns includes adding a legality using
    // markUnknownOpDynamicallyLegal for BranchOpInterface, so we override it
    // here. This is a bit hacky.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
