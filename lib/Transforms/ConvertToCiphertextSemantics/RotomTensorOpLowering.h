#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_

#include <cstdint>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

class RotomTensorOpLowering {
 public:
  explicit RotomTensorOpLowering(const ContextAwareTypeConverter* typeConverter)
      : typeConverter(typeConverter) {}

  LogicalResult lowerMatmul(
      linalg::MatmulOp op, linalg::MatmulOp::Adaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const;

  LogicalResult lowerElementwiseBinary(
      Operation* op, Value originalResult, ValueRange adaptorOperands,
      ContextAwareConversionPatternRewriter& rewriter) const;

 private:
  tensor_ext::LayoutAttr getLayoutAttr(Value value) const;

  FailureOr<std::vector<std::vector<int64_t>>> getRangePointsForDomain(
      tensor_ext::LayoutAttr layout, ArrayRef<int64_t> domain) const;

  FailureOr<Value> createMaskForPoints(
      RankedTensorType ciphertextSemanticType,
      const std::vector<std::vector<int64_t>>& points,
      ImplicitLocOpBuilder& b) const;

  FailureOr<DenseIntElementsAttr> createRemapAttr(
      MLIRContext* ctx, const std::vector<std::vector<int64_t>>& sourcePoints,
      const std::vector<std::vector<int64_t>>& targetPoints) const;

  FailureOr<Value> alignDomainPointToOutput(
      Value source, RankedTensorType ciphertextSemanticType,
      tensor_ext::LayoutAttr sourceLayout, ArrayRef<int64_t> sourceDomain,
      ArrayRef<int64_t> outputDomain, tensor_ext::LayoutAttr outputLayout,
      ImplicitLocOpBuilder& b) const;

  // Attempts the rotate-multiply-accumulate matmul kernel: groups every scalar
  // contribution by the ciphertext rotation that realizes it and emits one
  // masked rotate-product per group. Returns success (having replaced `op`)
  // when applicable, or failure (emitting nothing) so the caller can fall back
  // to the brute-force per-scalar lowering. Currently limited to single-
  // ciphertext layouts with injective slot packings.
  LogicalResult lowerMatmulByRotation(
      linalg::MatmulOp op, Value lhs, Value rhs, Value output,
      RankedTensorType ciphertextSemanticType, tensor_ext::LayoutAttr lhsLayout,
      tensor_ext::LayoutAttr rhsLayout, tensor_ext::LayoutAttr outputLayout,
      int64_t m, int64_t n, int64_t p,
      ContextAwareConversionPatternRewriter& rewriter) const;

  Value createRotate(Value tensor, int64_t shift, ImplicitLocOpBuilder& b) const;

  const ContextAwareTypeConverter* typeConverter;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
