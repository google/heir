#include "lib/Utils/TensorUtils.h"

#include <cstdint>

#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

FailureOr<int64_t> getFlattenedIndex(RankedTensorType tensorType,
                                     SmallVector<OpFoldResult> indices) {
  // Collect the constant indices into the tensor.
  auto maybeConstIndices = getConstantIntValues(indices);
  if (!maybeConstIndices.has_value()) return failure();
  auto constIndices =
      llvm::map_to_vector(maybeConstIndices.value(),
                          [](int64_t i) { return static_cast<uint64_t>(i); });

  auto rank = tensorType.getRank();
  int flatIndex = 0;
  int stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    flatIndex += constIndices[i] * stride;
    stride *= tensorType.getDimSize(i);
  }
  return flatIndex;
}

}  // namespace heir
}  // namespace mlir
