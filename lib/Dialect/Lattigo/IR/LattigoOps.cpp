#include "lib/Dialect/Lattigo/IR/LattigoOps.h"

#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

LogicalResult BGVEncodeOp::verify() {
  if (!isa<RankedTensorType>(getValue().getType())) {
    return emitError("value must be a ranked tensor");
  }
  return success();
}

LogicalResult BGVDecodeOp::verify() {
  if (getValue().getType() != getDecoded().getType()) {
    return emitError("value and decoded types must match");
  }
  if (!isa<RankedTensorType>(getDecoded().getType())) {
    return emitError("decoded must be a ranked tensor");
  }
  return success();
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
