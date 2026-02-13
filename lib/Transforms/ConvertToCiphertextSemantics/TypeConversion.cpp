#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;
using tensor_ext::LayoutAttr;

Type materializeLayout(Type dataType, LayoutAttr attr, int ciphertextSize) {
  IntegerRelation rel = attr.getIntegerRelation();
  llvm::SmallVector<int64_t> ciphertextSemanticShape;
  for (unsigned varPos = rel.getVarKindOffset(VarKind::Range);
       varPos < rel.getVarKindEnd(VarKind::Range) - 1; ++varPos) {
    std::optional<int64_t> dimBound =
        rel.getConstantBound64(BoundType::UB, varPos);
    assert(dimBound && "No upper bound found for range variable");
    ciphertextSemanticShape.push_back(dimBound.value() +
                                      1);  // +1 is because UB is inclusive
  }
  // Last dimension is always the slot size. The relation may enforce a tighter
  // bound depending on whether the slots at the end are full, so use the upper
  // bound.
  ciphertextSemanticShape.push_back(ciphertextSize);
  return RankedTensorType::get(ciphertextSemanticShape, dataType);
}

Type materializeScalarLayout(Type type, LayoutAttr attr, int ciphertextSize) {
  return RankedTensorType::get({1, ciphertextSize}, type);
}

Type materializePermutationLayout(Type elementType,
                                  DenseIntElementsAttr permutation,
                                  int ciphertextSize) {
  // TODO(#2666): Extend to a more general case where src_ct and dst_ct != 0
  // src_ct and dst_ct are always 0; output is always a single ciphertext.
  return RankedTensorType::get({1, ciphertextSize}, elementType);
}

}  // namespace heir
}  // namespace mlir
