#include "lib/Utils/Layout/Utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

// ISL
#include "include/isl/ctx.h"       // from @isl
#include "include/isl/map.h"       // from @isl
#include "include/isl/map_type.h"  // from @isl
#include "include/isl/point.h"     // from @isl
#include "include/isl/set.h"       // from @isl
#include "include/isl/space.h"     // from @isl
#include "include/isl/val.h"       // from @isl

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
  for (int i = 0; i < result.getVarKindOffset(VarKind::Local); ++i) {
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
  result.addBound(BoundType::UB, rangeOffset,
                  std::ceil((float)tensorType.getNumElements() / numSlots) - 1);
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
    RankedTensorType matrixType, int64_t ciphertextSize) {
  unsigned int rows = matrixType.getDimSize(0);
  unsigned int cols = matrixType.getDimSize(1);

  // The diagonals of the result must be able to fit an entire diagonal of the
  // matrix, so ensure that the number of columns (diagonal size) is less than
  // the result's columns.
  assert(cols <= ciphertextSize);

  // The number of rows must divide the number of columns.
  int64_t paddedCols = isPowerOfTwo(cols) ? cols : nextPowerOfTwo(cols);
  int64_t paddedRows = isPowerOfTwo(rows) ? rows : nextPowerOfTwo(rows);

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      matrixType.getRank(), /*numRange=*/2, /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for the data matrix dimensions.
  for (int i = 0; i < matrixType.getRank(); ++i) {
    result.addBound(BoundType::UB, i, matrixType.getDimSize(i) - 1);
    result.addBound(BoundType::LB, i, 0);
  }
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  for (int i = 0; i < 2; ++i) {
    result.addBound(BoundType::LB, rangeOffset + i, 0);
  }
  result.addBound(BoundType::UB, rangeOffset, paddedRows - 1);
  result.addBound(BoundType::UB, rangeOffset + 1, ciphertextSize - 1);

  // Add diagonal layout constraints:
  // slot % padded_rows = row
  SmallVector<int64_t> slotModCoeffs(result.getNumCols(), 0);
  slotModCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto slotMod = addModConstraint(result, slotModCoeffs, paddedRows);
  SmallVector<int64_t> slotEquality(result.getNumCols(), 0);
  slotEquality[result.getVarKindOffset(VarKind::Domain)] = 1;
  slotEquality[slotMod] = -1;
  result.addEquality(slotEquality);

  // (ct + slot) % padded_cols = col
  SmallVector<int64_t> ctSlotCoeffs(result.getNumCols(), 0);
  ctSlotCoeffs[result.getVarKindOffset(VarKind::Range)] = 1;
  ctSlotCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto ctSlotMod = addModConstraint(result, ctSlotCoeffs, paddedCols);
  SmallVector<int64_t> ctSlotEquality(result.getNumCols(), 0);
  ctSlotEquality[result.getVarKindOffset(VarKind::Domain) + 1] = 1;
  ctSlotEquality[ctSlotMod] = -1;
  result.addEquality(ctSlotEquality);

  return result;
}

presburger::IntegerRelation getPerRowLayoutRelation(RankedTensorType matrixType,
                                                    int64_t ciphertextSize) {
  auto domainSize = matrixType.getRank();
  assert(domainSize == 2 && "expected 2-D matrix");
  assert(matrixType.getDimSize(1) <= ciphertextSize &&
         "expected ciphertextSize >= matrixType.getDimSize(1)");

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      domainSize, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/0));

  // Add bounds for the matrix dimensions.
  for (int i = 0; i < matrixType.getRank(); ++i) {
    result.addBound(BoundType::UB, i, matrixType.getDimSize(i) - 1);
    result.addBound(BoundType::LB, i, 0);
  }
  // Number of ciphertexts is the number of rows.
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  result.addBound(BoundType::LB, rangeOffset, 0);
  result.addBound(BoundType::UB, rangeOffset, matrixType.getDimSize(0) - 1);
  result.addBound(BoundType::LB, rangeOffset + 1, 0);
  result.addBound(BoundType::UB, rangeOffset + 1, ciphertextSize - 1);

  // 0 = -rows + ct
  SmallVector<int64_t, 8> ctConstraint(result.getNumCols(), 0);
  ctConstraint[result.getVarKindOffset(VarKind::Domain)] = -1;
  ctConstraint[result.getVarKindOffset(VarKind::Range)] = 1;
  result.addEquality(ctConstraint);

  // The slotMod = slot % nextPowerOfTwo(cols)
  auto paddedCols = nextPowerOfTwo(matrixType.getDimSize(1));
  SmallVector<int64_t> slotCoeffs(result.getNumCols(), 0);
  slotCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto slotMod = addModConstraint(result, slotCoeffs, paddedCols);

  // slotMod - col = 0
  SmallVector<int64_t> eqConstraint(result.getNumCols(), 0);
  eqConstraint[slotMod] = 1;
  eqConstraint[result.getVarKindOffset(VarKind::Domain) + 1] = -1;
  result.addEquality(eqConstraint);

  return result;
}

bool isRelationSquatDiagonal(RankedTensorType matrixType,
                             int64_t ciphertextSize,
                             const presburger::IntegerRelation& relation) {
  IntegerRelation diagonalRelation =
      getDiagonalLayoutRelation(matrixType, ciphertextSize);
  return relation.isEqual(diagonalRelation);
}

bool isRelationRowMajor(RankedTensorType vectorType, int64_t numSlots,
                        const presburger::IntegerRelation& relation) {
  IntegerRelation rowMajorRelation =
      getRowMajorLayoutRelation(vectorType, numSlots);
  return relation.isEqual(rowMajorRelation);
}

bool isRelationPerRow(RankedTensorType matrixType, int64_t ciphertextSize,
                      presburger::IntegerRelation relation) {
  IntegerRelation perRowRelation =
      getPerRowLayoutRelation(matrixType, ciphertextSize);
  return relation.isEqual(perRowRelation);
}

presburger::IntegerRelation collapseDimensions(
    const presburger::IntegerRelation& relation, RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociation) {
  std::unique_ptr<IntegerRelation> clonedRelation = relation.clone();
  for (const ReassociationIndices& associationGroup : reassociation) {
    // a single-entry association group is a no-op
    if (associationGroup.size() == 1) {
      continue;
    }
    for (int64_t reassocDim : associationGroup) {
      if (sourceType.getShape()[reassocDim] == 1) {
        // Drop this unit dimension
        clonedRelation->setAndEliminate(reassocDim, 0);
      }
    }
  }
  return std::move(*clonedRelation);
}

presburger::IntegerRelation expandDimensions(
    const presburger::IntegerRelation& relation, RankedTensorType resultType,
    ArrayRef<ReassociationIndices> reassociation) {
  // tensor indices correspond to layout dimensions, and adding a dimension of
  // size 1 has no effect on the affine map expressions, so all we're doing is
  // adding new dimensions for each reassociation group index corresponding to
  // an output dimension of size 1. Mainly we have to ensure that the
  // dimension we're adding is in the correct index of the integer relations
  // domain variable list.
  std::unique_ptr<IntegerRelation> clonedRelation = relation.clone();
  int oldDim = 0;
  DenseMap<AffineExpr, AffineExpr> oldDimsToNewDims;
  for (const ReassociationIndices& associationGroup : reassociation) {
    // a single-entry association group is a no-op
    if (associationGroup.size() == 1) {
      ++oldDim;
      continue;
    }

    for (int64_t reassocDim : associationGroup) {
      if (resultType.getShape()[reassocDim] > 1) {
        ++oldDim;
      } else {
        // A new dimension of size 1 is being added, so add a new domain
        // variable v with 0 <= v < 1.
        auto newDimIndex = clonedRelation->insertVar(VarKind::Domain, oldDim);
        clonedRelation->addBound(BoundType::LB, newDimIndex, 0);
        clonedRelation->addBound(BoundType::UB, newDimIndex, 0);
        ++oldDim;
      }
    }
  }
  return std::move(*clonedRelation);
}

presburger::IntegerRelation fixVars(const presburger::IntegerRelation& relation,
                                    ArrayRef<int64_t> fixedValues,
                                    presburger::VarKind varKind) {
  std::unique_ptr<IntegerRelation> rel = relation.clone();

  // One constraint for each fixed variable
  for (auto [dim, value] : llvm::enumerate(fixedValues)) {
    SmallVector<int64_t> constraint(relation.getNumCols(), 0);
    constraint[dim + relation.getVarKindOffset(varKind)] = 1;
    constraint.back() = -value;
    rel->addEquality(constraint);
  }

  rel->simplify();
  rel->removeRedundantConstraints();
  return std::move(*rel);
}

isl_stat pointCallback(__isl_take isl_point* pnt, void* user) {
  PointCollector* collector = static_cast<PointCollector*>(user);

  // Use isl_space_dim instead of accessing struct members directly
  isl_space* space = isl_point_get_space(pnt);
  int dim = isl_space_dim(space, isl_dim_set);
  isl_space_free(space);

  std::vector<int64_t> point(dim);

  for (int i = 0; i < dim; i++) {
    isl_val* coord = isl_point_get_coordinate_val(pnt, isl_dim_set, i);
    if (isl_val_is_int(coord)) {
      point[i] = isl_val_get_num_si(coord);
    }
    isl_val_free(coord);
  }

  collector->points.push_back(point);
  isl_point_free(pnt);
  return isl_stat_ok;
}

void getRangePoints(const presburger::IntegerRelation& relation,
                    PointCollector& collector) {
  auto* bmap = convertRelationToBasicMap(relation, collector.ctx);
  isl_set* set = isl_set_from_basic_set(isl_basic_map_range(bmap));
  isl_set_foreach_point(set, &pointCallback, &collector);
  isl_set_free(set);
}

std::vector<int64_t> anyRangePoint(
    const presburger::IntegerRelation& relation) {
  isl_ctx* ctx = isl_ctx_alloc();
  auto* bmap = convertRelationToBasicMap(relation, ctx);
  isl_basic_set* bset = isl_basic_map_range(bmap);
  isl_point* point = isl_basic_set_sample_point(bset);

  if (!point) {
    return {};
  }

  isl_space* space = isl_point_get_space(point);
  int dim = isl_space_dim(space, isl_dim_set);
  isl_space_free(space);
  std::vector<int64_t> result;
  result.reserve(dim);
  for (int i = 0; i < dim; i++) {
    isl_val* coord = isl_point_get_coordinate_val(point, isl_dim_set, i);
    if (isl_val_is_int(coord)) {
      result.push_back(isl_val_get_num_si(coord));
    }
    isl_val_free(coord);
  }
  isl_point_free(point);
  isl_ctx_free(ctx);

  return result;
}

}  // namespace heir
}  // namespace mlir
