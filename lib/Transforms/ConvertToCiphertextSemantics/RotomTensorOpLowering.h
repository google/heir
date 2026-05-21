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

  const ContextAwareTypeConverter* typeConverter;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
