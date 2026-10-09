#include "lib/Utils/TensorUtils.h"

#include <cassert>
#include <cstdint>

#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/IndexingUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

int64_t getFlattenedIndex(RankedTensorType tensorType,
                          SmallVector<int64_t> indices) {
  return mlir::linearize(indices, computeSuffixProduct(tensorType.getShape()));
}

FailureOr<int64_t> getFlattenedIndex(RankedTensorType tensorType,
                                     SmallVector<OpFoldResult> indices) {
  // Collect the constant indices into the tensor.
  auto maybeConstIndices = getConstantIntValues(indices);
  if (!maybeConstIndices.has_value()) return failure();
  auto constIndices = maybeConstIndices.value();

  return getFlattenedIndex(tensorType, constIndices);
}

SmallVector<int64_t> getIndicesFromRowMajorShape(int64_t flattenedIndex,
                                                 SmallVector<int64_t> shape) {
  return mlir::delinearize(flattenedIndex, computeSuffixProduct(shape));
}

SmallVector<ReassociationIndices> getReassociationForReshapeAtDim(
    int64_t rank, ArrayRef<int64_t> positions) {
  SmallVector<ReassociationIndices> reassociation;
  reassociation.reserve(rank - positions.size());

  llvm::DenseMap<int64_t, bool> positionsMap;
  for (int64_t pos : positions) {
    positionsMap[pos] = true;
  }
  auto isUnitDim = [&](int64_t dim) { return positionsMap.contains(dim); };

  ReassociationIndices reassociationGroup;
  unsigned dim = 0;
  while (dim < rank && isUnitDim(dim)) reassociationGroup.push_back(dim++);
  while (dim < rank) {
    assert(!isUnitDim(dim) && "expected non unit-extent");
    reassociationGroup.push_back(dim);
    ++dim;
    // Fold all following dimensions that are unit-extent.
    while (dim < rank && isUnitDim(dim)) {
      reassociationGroup.push_back(dim++);
    }
    reassociation.push_back(reassociationGroup);
    reassociationGroup.clear();
  }
  return reassociation;
}

}  // namespace heir
}  // namespace mlir
