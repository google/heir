#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_

#include <cstdint>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
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

  LogicalResult lowerElementwiseBinary(
      Operation* op, Value originalResult, ValueRange adaptorOperands,
      ContextAwareConversionPatternRewriter& rewriter) const;

 private:
  tensor_ext::LayoutAttr getLayoutAttr(Value value) const;

  FailureOr<std::vector<std::vector<int64_t>>> getRangePointsForDomain(
      tensor_ext::LayoutAttr layout, ArrayRef<int64_t> domain) const;

  // Same as above but reuses an already-built IntegerRelation. Building the
  // relation (LayoutAttr::getIntegerRelation serializes the layout to an ISL
  // string and re-parses it) is the dominant cost, so a caller that queries
  // many domain points of one layout builds the relation once and reuses it
  // here, turning an O(m*n) ISL rebuild into O(1).
  FailureOr<std::vector<std::vector<int64_t>>> getRangePointsForDomain(
      const presburger::IntegerRelation& relation,
      ArrayRef<int64_t> domain) const;

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
