#include "lib/Dialect/Cheddar/IR/CheddarOps.h"

#include <algorithm>

#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Utils/RotationUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

::llvm::SmallVector<::mlir::OpFoldResult> HRotOp::getRotationIndices() {
  if (getStaticDistance()) return {getStaticDistanceAttr()};
  return {getDynamicDistance()};
}

LogicalResult HRotOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicDistance(),
                                       getStaticDistance());
}

::llvm::SmallVector<::mlir::OpFoldResult> HRotAddOp::getRotationIndices() {
  return {getDistanceAttr()};
}

::llvm::SmallVector<::mlir::OpFoldResult>
LinearTransformOp::getRotationIndices() {
  auto diagonalsType = cast<ShapedType>(getDiagonals().getType());
  int64_t slots = diagonalsType.getShape()[1];
  auto rotations = lintransRotationIndices(
      getDiagonalIndicesAttr().asArrayRef(), slots, getBs().getInt());
  SmallVector<OpFoldResult> result;
  result.reserve(rotations.size());
  auto* mlirCtx = (*this)->getContext();
  for (int64_t rot : rotations) {
    result.push_back(IntegerAttr::get(IndexType::get(mlirCtx), rot));
  }
  return result;
}
LogicalResult LinearTransformOp::verify() {
  auto diagonalsType = cast<ShapedType>(getDiagonals().getType());
  if (diagonalsType.getRank() != 2) {
    return emitOpError("diagonals must be a 2D tensor or memref");
  }
  if (diagonalsType.getShape()[0] !=
      getDiagonalIndicesAttr().asArrayRef().size()) {
    return emitOpError(
        "number of diagonals must match number of diagonal indices");
  }
  return success();
}

LogicalResult EvalPolyOp::verify() { return success(); }

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
