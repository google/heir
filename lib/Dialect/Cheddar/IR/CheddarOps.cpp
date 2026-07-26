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
  // diagonals is TensorOrMemRef and is bufferized to a memref downstream, so
  // match on ShapedType rather than RankedTensorType (which would crash post
  // bufferization). The slot count is the row width (second dim).
  auto diagonalsType = cast<ShapedType>(getDiagonals().getType());
  int64_t slots = diagonalsType.getShape()[1];
  // Derive the key set from the op's baby-step count, the same value the
  // emitter hands to CHEDDAR, so key generation and evaluation agree.
  auto rotations = lintransRotationIndicesWithBabyStep(
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
  // `diagonals` is the matrix as a set of non-zero diagonals: one row per
  // diagonal, each row `slots` wide. getRotationIndices() and the emitter both
  // index getShape()[1], so a non-2D operand must be rejected here rather than
  // crashing them.
  auto diagonalsType = cast<ShapedType>(getDiagonals().getType());
  if (diagonalsType.getRank() != 2)
    return emitOpError(
               "expected `diagonals` to be 2D (one row per diagonal), but got "
               "rank ")
           << diagonalsType.getRank();

  auto indices = getDiagonalIndicesAttr().asArrayRef();
  int64_t numRows = diagonalsType.getShape()[0];
  if (numRows != static_cast<int64_t>(indices.size()))
    return emitOpError("expected one `diagonal_indices` entry per `diagonals` "
                       "row, but got ")
           << indices.size() << " indices for " << numRows << " rows";

  int64_t bs = getBs().getInt();
  int64_t gs = getGs().getInt();
  if (bs < 1 || gs < 1)
    return emitOpError("expected `bs` and `gs` to be >= 1, but got bs=")
           << bs << " gs=" << gs;

  // The BSGS decomposition only reaches diagonals within the `bs * gs` grid;
  // anything past it would be silently dropped. (Indices are non-negative,
  // with wrap-around diagonals encoded as `slot - k`.)
  if (!indices.empty()) {
    int32_t maxIdx = *std::max_element(indices.begin(), indices.end());
    if (bs * gs <= maxIdx)
      return emitOpError("BSGS grid `bs * gs` (")
             << (bs * gs) << ") must exceed the largest diagonal index ("
             << maxIdx << ") so the decomposition covers every diagonal";
  }
  return success();
}

LogicalResult EvalPolyOp::verify() {
  if (getCoefficients().empty())
    return emitOpError("expected a non-empty `coefficients` array");

  int64_t level = getLevel().getInt();
  int64_t outputLevel = getOutputLevel().getInt();
  if (level < 0 || outputLevel < 0)
    return emitOpError(
               "expected non-negative `level`/`outputLevel`, but got level=")
           << level << " outputLevel=" << outputLevel;
  // EvalPoly consumes multiplicative depth, so the result lands at or below the
  // input level -- it can never raise it.
  if (outputLevel > level)
    return emitOpError("expected `outputLevel` (")
           << outputLevel << ") <= `level` (" << level
           << "): the polynomial consumes depth, it cannot raise the level";
  return success();
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
