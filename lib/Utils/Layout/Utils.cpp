#include "lib/Utils/Layout/Utils.h"

#include <cassert>
#include <cstdint>

#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

// Adds a modulo constraint to the result relation. Returns the index of the new
// local variable that represents the modulo operation result.
unsigned int addModConstraint(IntegerRelation& result, ArrayRef<int64_t> exprs,
                              int64_t modulus) {
  assert(modulus > 0 && "addModConstraint modulus argument must be positive");

  // Add a local variable for the quotient q, i.e., expr % c is replaced by
  // (expr - q * c) where q = expr floordiv c.
  result.addLocalFloorDiv(exprs, modulus);

  // Add equality: mod = expr - q * c
  auto modIndex = result.appendVar(VarKind::Local);
  SmallVector<int64_t> modCoeffs(result.getNumCols(), 0);
  for (int i = 0; i < exprs.size() - 1; ++i) {
    modCoeffs[i] = exprs[i];
  }
  modCoeffs.back() = exprs.back();
  auto lastLocal = result.getVarKindEnd(VarKind::Local) - 1;
  modCoeffs[lastLocal - 1] = -modulus;  // -q * c
  modCoeffs[lastLocal] = -1;            // -mod
  result.addEquality(modCoeffs);

  return modIndex;
}

presburger::IntegerRelation getRowMajorLayoutRelation(
    RankedTensorType tensorType, int64_t numSlots) {
  auto domainSize = tensorType.getRank();
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      domainSize, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/0));

  // Add bounds for the matrix dimensions.
  for (int i = 0; i < tensorType.getRank(); ++i) {
    result.addBound(BoundType::UB, i, tensorType.getDimSize(i) - 1);
    result.addBound(BoundType::LB, i, 0);
  }
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  result.addBound(BoundType::LB, rangeOffset, 0);
  result.addBound(BoundType::LB, rangeOffset + 1, 0);
  result.addBound(BoundType::UB, rangeOffset + 1, numSlots - 1);

  // 0 = (flattened_expr) floordiv ciphertextSize - ct
  // We first need to add a local var q to represent the floordiv and then add
  // the equality with ct to compute the ciphertext index.
  // Get row-major layout expression.
  SmallVector<int64_t> rowMajorCoeffs(result.getNumCols(), 0);
  unsigned product = 1;
  for (int dim = result.getVarKindEnd(VarKind::Domain) - 1;
       dim >= (int)result.getVarKindOffset(VarKind::Domain); --dim) {
    rowMajorCoeffs[dim] = product;
    product *= tensorType.getDimSize(dim);
  }
  // q = flattened_expr floordiv numSlots
  result.addLocalFloorDiv(rowMajorCoeffs, numSlots);
  // 0 = q - ct
  SmallVector<int64_t, 8> ctConstraint(result.getNumCols(), 0);
  ctConstraint[result.getVarKindOffset(VarKind::Range)] = -1;
  ctConstraint[result.getVarKindEnd(VarKind::Local) - 1] = 1;
  result.addEquality(ctConstraint);

  // The next constraint computes the slot index assuming the domain
  // size is a power of two. This is required to ensure cyclic rotations
  // are consistent when data is smaller than the total number of slots
  // in ciphertext. We do this in three steps:
  // 1. flattened_expr mod numSlots = b
  // 2. slot mod paddedSize = a
  // 3. a = b

  // First, we need to insert a new local variable (q) into the row major
  // constraint.
  SmallVector<int64_t> flattenedCoeffs(result.getNumCols(), 0);
  for (int i = 0; i < rowMajorCoeffs.size(); i++) {
    flattenedCoeffs[i] = rowMajorCoeffs[i];
  }
  // flattened_expr mod numSlots = b
  auto rhsMod = addModConstraint(result, flattenedCoeffs, numSlots);

  // slot mod paddedSize = a
  int64_t paddedSize = isPowerOfTwo(tensorType.getNumElements())
                           ? tensorType.getNumElements()
                           : nextPowerOfTwo(tensorType.getNumElements());
  SmallVector<int64_t> slotModCoeffs(result.getNumCols(), 0);
  slotModCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto lhsMod = addModConstraint(result, slotModCoeffs, paddedSize);

  // a = b
  SmallVector<int64_t> eqConstraint(result.getNumCols(), 0);
  eqConstraint[rhsMod] = 1;
  eqConstraint[lhsMod] = -1;
  result.addEquality(eqConstraint);

  return result;
}

presburger::IntegerRelation getDiagonalLayoutRelation(
    RankedTensorType matrixType, RankedTensorType diagonalizedType) {
  unsigned int rows = matrixType.getDimSize(0);
  unsigned int cols = matrixType.getDimSize(1);

  assert(rows == diagonalizedType.getDimSize(0));
  // Number of rows must be less than or equal to the number of columns.
  assert(rows <= cols);
  // The diagonals of the result must be able to fit an entire diagonal of the
  // matrix, so ensure that the number of columns (diagonal size) is less than
  // the result's columns.
  assert(cols <= diagonalizedType.getDimSize(1));
  assert(diagonalizedType.getRank() == 2);

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      matrixType.getRank(), /*numRange=*/2, /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for the matrix dimensions.
  for (int i = 0; i < matrixType.getRank(); ++i) {
    result.addBound(BoundType::UB, i, matrixType.getDimSize(i) - 1);
    result.addBound(BoundType::LB, i, 0);
  }
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  for (int i = 0; i < diagonalizedType.getRank(); ++i) {
    result.addBound(BoundType::UB, rangeOffset + i,
                    diagonalizedType.getDimSize(i) - 1);
    result.addBound(BoundType::LB, rangeOffset + i, 0);
  }

  // Add diagonal layout constraints:
  // slot % rows = row
  SmallVector<int64_t> slotModCoeffs(result.getNumCols(), 0);
  slotModCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto slotMod = addModConstraint(result, slotModCoeffs, rows);
  SmallVector<int64_t> slotEquality(result.getNumCols(), 0);
  slotEquality[result.getVarKindOffset(VarKind::Domain)] = 1;
  slotEquality[slotMod] = -1;
  result.addEquality(slotEquality);

  // (ct + slot) % cols = col
  SmallVector<int64_t> ctSlotCoeffs(result.getNumCols(), 0);
  ctSlotCoeffs[result.getVarKindOffset(VarKind::Range)] = 1;
  ctSlotCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto ctSlotMod = addModConstraint(result, ctSlotCoeffs, cols);
  SmallVector<int64_t> ctSlotEquality(result.getNumCols(), 0);
  ctSlotEquality[result.getVarKindOffset(VarKind::Domain) + 1] = 1;
  ctSlotEquality[ctSlotMod] = -1;
  result.addEquality(ctSlotEquality);

  result.simplify();
  return result;
}

bool isRelationSquatDiagonal(RankedTensorType matrixType,
                             RankedTensorType ciphertextSemanticShape,
                             presburger::IntegerRelation relation) {
  IntegerRelation diagonalRelation =
      getDiagonalLayoutRelation(matrixType, ciphertextSemanticShape);
  return relation.isEqual(diagonalRelation);
}

}  // namespace heir
}  // namespace mlir
