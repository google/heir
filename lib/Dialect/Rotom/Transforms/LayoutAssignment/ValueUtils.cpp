#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/ValueUtils.h"

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/IR/Attributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir::heir::rotom {

Type getPlainValueType(Type type) {
  if (auto secretType = dyn_cast<secret::SecretType>(type)) {
    return secretType.getValueType();
  }
  return type;
}

bool isTensorLike(Value value) {
  return isa<RankedTensorType>(getPlainValueType(value.getType()));
}

bool isLayoutCompatibleWithValue(LayoutAttr layout, Value value) {
  auto type = dyn_cast<RankedTensorType>(getPlainValueType(value.getType()));
  if (!type) return false;

  int64_t rank = type.getRank();
  for (Attribute attr : layout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) continue;
    int64_t dimIndex = dim.getDim();
    if (dimIndex >= rank) return false;
    int64_t typeDimSize = type.getDimSize(dimIndex);
    if (typeDimSize == ShapedType::kDynamic) continue;
    if (typeDimSize <= 0) continue;
    int64_t paddedDimSize = nextPowerOfTwo(typeDimSize);
    if (dim.getSize() * dim.getStride() > paddedDimSize) return false;
  }
  return true;
}

}  // namespace mlir::heir::rotom
