#include "lib/Transforms/ConvertToCiphertextSemantics/TensorKernelSupport.h"

#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"      // from @llvm-project

namespace mlir::heir {
LogicalResult checkEncryptedExtract(tensor::ExtractOp op) {
  // A runtime index would require dynamic rotations and a corresponding key
  // selection strategy when moving the selected slot into the result layout.
  // TODO(#2257): Support dynamic indices in tensor.extract.
  if (getConstantIntValues(getAsOpFoldResult(op.getIndices())).has_value())
    return success();
  auto diag = op.emitError(
      "indexing an encrypted tensor with a runtime index "
      "is not supported by the tensor-to-ciphertext lowering");
  diag.attachNote() << "tensor.extract requires compile-time constant indices; "
                       "use a constant index if appropriate for this model";
  return failure();
}

LogicalResult checkEncryptedPad(tensor::PadOp op) {
  if (!op.getSourceType().hasStaticShape() ||
      !op.getResultType().hasStaticShape()) {
    auto diag = op.emitError(
        "padding an encrypted tensor requires static input "
        "and output shapes");
    diag.attachNote() << "export the model with fixed tensor dimensions and "
                         "compile-time constant padding amounts";
    return failure();
  }
  for (auto amounts : {op.getStaticLow(), op.getStaticHigh()}) {
    for (int64_t amount : amounts) {
      if (ShapedType::isDynamic(amount)) {
        auto diag = op.emitError(
            "runtime padding amounts for encrypted tensors "
            "are not supported");
        diag.attachNote()
            << "use compile-time constant low and high padding amounts";
        return failure();
      }
    }
  }
  Value padding =
      cast<tensor::YieldOp>(op.getRegion().front().getTerminator()).getValue();
  if (!matchPattern(padding, m_AnyZeroFloat()) &&
      !matchPattern(padding, m_Zero())) {
    auto diag = op.emitError(
        "padding an encrypted tensor with a nonzero or "
        "runtime value is not supported");
    diag.attachNote()
        << "this lowering supports constant zero padding; use zero "
           "padding if appropriate for this model";
    return failure();
  }
  return success();
}
}  // namespace mlir::heir
