#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;
using tensor_ext::LayoutAttr;

Type materializeLayout(Type dataType, LayoutAttr attr, int ciphertextSize) {
  IntegerRelation rel = attr.getIntegerRelation();
  llvm::SmallVector<int64_t> ciphertextSemanticShape;
  unsigned rangeOffset = rel.getVarKindOffset(VarKind::Range);
  for (unsigned varPos = rangeOffset;
       varPos < rel.getVarKindEnd(VarKind::Range) - 1; ++varPos) {
    LLVM_DEBUG({
      llvm::dbgs() << "materializeLayout: computing upper bound for range "
                      "dimension "
                   << varPos - rangeOffset << " (ct), layout=" << attr << "\n";
      llvm::dbgs().flush();
    });
    std::optional<int64_t> dimBound =
        rel.getConstantBound64(BoundType::UB, varPos);
    LLVM_DEBUG({
      llvm::dbgs() << "materializeLayout: upper bound for range dimension "
                   << varPos - rangeOffset << " (ct) = ";
      if (dimBound) {
        llvm::dbgs() << *dimBound;
      } else {
        llvm::dbgs() << "none";
      }
      llvm::dbgs() << "\n";
    });
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

Type materializePermutationLayout(Type type, DenseIntElementsAttr permutation,
                                  int ciphertextSize) {
  auto tensorType = dyn_cast<RankedTensorType>(type);

  assert(tensorType &&
         "Permutation layout attributes on non-tensor args are not supported");
  assert(tensorType.getShape().size() <= 2 &&
         "Permutation layouts only supports tensor args of max dim-2");

  auto inShape = tensorType.getShape();
  if (inShape.size() == 2)
    return RankedTensorType::get({inShape[0], ciphertextSize},
                                 getElementTypeOrSelf(type));

  return RankedTensorType::get({1, ciphertextSize}, getElementTypeOrSelf(type));
}

}  // namespace heir
}  // namespace mlir
