#include "lib/Dialect/Cheddar/IR/CheddarOps.h"

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
  if (getStaticShift()) return {getStaticShiftAttr()};
  return {getDynamicShift()};
}

LogicalResult HRotOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicShift(),
                                       getStaticShift());
}

::llvm::SmallVector<::mlir::OpFoldResult> HRotAddOp::getRotationIndices() {
  return {getDistanceAttr()};
}

::llvm::SmallVector<::mlir::OpFoldResult>
LinearTransformOp::getRotationIndices() {
  auto diagonalsType = cast<RankedTensorType>(getDiagonals().getType());
  int64_t slots = diagonalsType.getShape()[1];
  int64_t logBSGS = getLogBabyStepGiantStepRatio().getInt();
  auto rotations = lintransRotationIndices(
      getDiagonalIndicesAttr().asArrayRef(), slots, logBSGS);
  SmallVector<OpFoldResult> result;
  result.reserve(rotations.size());
  auto* mlirCtx = (*this)->getContext();
  for (int64_t rot : rotations) {
    result.push_back(IntegerAttr::get(IndexType::get(mlirCtx), rot));
  }
  return result;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
