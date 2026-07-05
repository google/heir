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

  // Brings a ciphertext-semantic value packed as `fromLayout` to `toLayout`.
  // Same ciphertext count: one tensor_ext.convert_layout (lowered later by
  // the shift network). Different ciphertext count -- which convert_layout
  // cannot express -- the explicit rotate/mask/accumulate steps of
  // planLayoutExpansion (both expansion and compaction), exactly as the
  // layout assignment priced them. Returns a null Value on failure.
  Value convertToLayout(ImplicitLocOpBuilder& b, Value value,
                        tensor_ext::LayoutAttr fromLayout,
                        tensor_ext::LayoutAttr toLayout) const;

 private:
  tensor_ext::LayoutAttr getLayoutAttr(Value value) const;

  const ContextAwareTypeConverter* typeConverter;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
