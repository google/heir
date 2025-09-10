#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Utils.h"
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
using tensor_ext::NewLayoutAttr;

Type materializeScalarLayout(Type type, LayoutAttr attr, int ciphertextSize) {
  // TODO(#1662): improve scalar layout materialization
  // Support scalars with non-repetition layouts (e.g., in slot 0 with
  // 0-padding); currently the layout system always produces a pure-repeition
  // alignment and a trivial row-major layout.
  return RankedTensorType::get({ciphertextSize}, type);
}

Type materializeLayout(RankedTensorType type, LayoutAttr attr,
                       int ciphertextSize) {
  AffineMap layout = attr.getMap();

  // First extract the tensor type as expanded according to the
  // alignment attribute.
  tensor_ext::AlignmentAttr alignment = attr.getAlignment();
  if (alignment) {
    type = RankedTensorType::get(alignment.getOut(), type.getElementType());
  }

  // Each ciphertext will always have ciphertextSize many slots, so the main
  // goal is to determine how many ciphertexts are needed. We do this by
  // iterating over the input type's index domain, and apply the layout
  // affine map to each index, and keep track of the maximum value of each
  // index of the map results. These maxima (plus 1 for zero indexing)
  // will be the shape of the new type.
  SmallVector<int64_t> outputTensorShape(layout.getNumResults(), 0);
  outputTensorShape[layout.getNumResults() - 1] = ciphertextSize;

  // Evaluate the affine map on the input indices and update the
  // outputTensorShape to be a max over visited indices.
  IndexTupleConsumer evaluateNextIndex =
      [&](const std::vector<int64_t>& indices) {
        SmallVector<int64_t> results;
        evaluateStatic(layout, indices, results);

        // minus 1 to skip the last dimension (ciphertext dimension)
        for (int i = 0; i < layout.getNumResults() - 1; ++i) {
          // 1 + to account for zero indexing
          outputTensorShape[i] = std::max(outputTensorShape[i], 1 + results[i]);
        }
      };

  iterateIndices(type.getShape(), evaluateNextIndex);
  return RankedTensorType::get(outputTensorShape, type.getElementType());
}

Type materializeNewLayout(Type dataType, NewLayoutAttr attr,
                          int ciphertextSize) {
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

}  // namespace heir
}  // namespace mlir
