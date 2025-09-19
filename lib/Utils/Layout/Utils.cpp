#include "lib/Utils/Layout/Utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
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

namespace {

// Helper that adds constraints built from the array of positions and coeffs.
// Inequalities are given by (>= 0).
void addConstraint(IntegerRelation& result,
                   ArrayRef<std::pair<int64_t, int64_t>> posAndCoeff,
                   bool equality) {
  SmallVector<int64_t> eqConstraint(result.getNumCols(), 0);
  for (auto [pos, coeff] : posAndCoeff) {
    assert(pos >= 0 && pos < result.getNumCols() && "invalid coeff position");
    eqConstraint[pos] = coeff;
  }
  if (equality) {
    result.addEquality(eqConstraint);
  } else {
    result.addInequality(eqConstraint);
  }
}

// Helper that adds inclusive lower and upper bounds for a given position and
// value.
void addBounds(IntegerRelation& result, int64_t pos, int64_t lower,
               std::optional<int64_t> upper = std::nullopt) {
  result.addBound(BoundType::LB, pos, lower);
  if (upper.has_value()) {
    result.addBound(BoundType::UB, pos, upper.value());
  }
}

}  // namespace

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
    addBounds(result, i, 0, tensorType.getDimSize(i) - 1);
  }
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  addBounds(result, rangeOffset, 0,
            std::ceil((float)tensorType.getNumElements() / numSlots) - 1);
  addBounds(result, rangeOffset + 1, 0, numSlots - 1);

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
  addConstraint(result,
                {{result.getVarKindOffset(VarKind::Range), -1},
                 {result.getVarKindEnd(VarKind::Local) - 1, 1}},
                /*equality=*/true);

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
    addBounds(result, i, 0, matrixType.getDimSize(i) - 1);
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
    addBounds(result, i, 0, matrixType.getDimSize(i) - 1);
  }
  // Number of ciphertexts is the number of rows.
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  addBounds(result, rangeOffset, 0, matrixType.getDimSize(0) - 1);
  addBounds(result, rangeOffset + 1, 0, ciphertextSize - 1);

  // 0 = -rows + ct
  addConstraint(result,
                {{result.getVarKindOffset(VarKind::Domain), -1},
                 {result.getVarKindOffset(VarKind::Range), 1}},
                /*equality=*/true);

  // The slotMod = slot % nextPowerOfTwo(cols)
  auto paddedCols = nextPowerOfTwo(matrixType.getDimSize(1));
  SmallVector<int64_t> slotCoeffs(result.getNumCols(), 0);
  slotCoeffs[result.getVarKindOffset(VarKind::Range) + 1] = 1;
  auto slotMod = addModConstraint(result, slotCoeffs, paddedCols);

  // slotMod - col = 0
  addConstraint(
      result,
      {{slotMod, 1}, {result.getVarKindOffset(VarKind::Domain) + 1, -1}},
      /*equality=*/true);

  return result;
}

presburger::IntegerRelation get2dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    int64_t padding) {
  auto domainSize = filterType.getRank();
  assert(domainSize == 2 && "expected 2-D filter matrix");

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      domainSize, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/2));

  // These are the indices that represent the position of the top left index of
  // the filter as it slides over the data.
  auto dataRow = result.getVarKindOffset(VarKind::Local);
  auto dataCol = result.getVarKindOffset(VarKind::Local) + 1;

  // Filter row and column indices
  auto filterRow = result.getVarKindOffset(VarKind::Domain);
  auto filterCol = result.getVarKindOffset(VarKind::Domain) + 1;

  // Matrix row and column indices
  auto matRow = result.getVarKindOffset(VarKind::Range);
  auto matCol = result.getVarKindOffset(VarKind::Range) + 1;

  // Constant coefficient
  auto constCoeff = result.getNumCols() - 1;

  // Filter and datasize
  auto filterRowSize = filterType.getDimSize(0);
  auto filterColSize = filterType.getDimSize(1);
  auto dataRowSize = dataType.getDimSize(0);
  auto dataColSize = dataType.getDimSize(1);

  auto filterSlidingRows = dataRowSize + 2 * padding - filterRowSize + 1;
  auto filterSlidingCols = dataColSize + 2 * padding - filterColSize + 1;

  // Add bounds for the filter matrix dimensions.
  addBounds(result, filterRow, 0, filterRowSize - 1);
  addBounds(result, filterCol, 0, filterColSize - 1);

  // Add bounds for the locals (the filter sliding indices).
  // These are indexed starting from -padding since they track the top left
  // corner of the filter matrix.
  addBounds(result, dataRow, -padding, filterSlidingRows - 1 - padding);
  addBounds(result, dataCol, -padding, filterSlidingCols - 1 - padding);

  // Add constraints for when the resulting filter index is in range.
  // 0 <= filterRow + dataRow < dataRowSize
  addConstraint(result, {{filterRow, 1}, {dataRow, 1}}, /*equality=*/false);
  addConstraint(result,
                {{constCoeff, dataRowSize - 1}, {filterRow, -1}, {dataRow, -1}},
                /*equality=*/false);

  // 0 <= filterCol + dataCol < dataColSize
  addConstraint(result, {{filterCol, 1}, {dataCol, 1}}, /*equality=*/false);
  addConstraint(result,
                {{constCoeff, dataColSize - 1}, {filterCol, -1}, {dataCol, -1}},
                /*equality=*/false);

  // Add equalities for the resulting matrix row and column. The matrix row
  // corresponds to the flattened data index (since it corresponds to one
  // iteration of the filter as it slides over the data) normalized by adding
  // the padding offset back to the indices.
  // The matrix column corresponds to the index into the filter plus the offsets
  // from the padding and the filter sliding iteration (the matrix row).
  //
  // fsr, fsc = filter sliding row size, col size
  // fr, fc = filter row size, col size
  // idr, idc = index of data row, col
  // ifr, ifc = index of filter row, col
  // mr, mc = matrix row, matrix col
  //
  // mr = (idr + P)*fsc + (idc + P) = P + P*fsc + idr * fsc + idc
  // mc = -(P + fc * P) + mr + (ifc) + fr*(ifr)
  addConstraint(result,
                {{matRow, -1},
                 {constCoeff, (filterSlidingCols + 1) * (padding)},
                 {dataRow, filterSlidingCols},
                 {dataCol, 1}},
                /*equality=*/true);
  addConstraint(result,
                {{matCol, -1},
                 {constCoeff, -(padding + filterColSize * padding)},
                 {matRow, 1},
                 {filterCol, 1},
                 {filterRow, filterRowSize}},
                /*equality=*/true);
  return result;
}

FailureOr<presburger::IntegerRelation> get2dConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize) {
  auto expandedFilterRelation =
      get2dConvFilterRelation(filterType, dataType, padding);
  // Get size of the expanded filter matrix.
  auto rowBound = expandedFilterRelation.getConstantBound64(
      BoundType::UB, expandedFilterRelation.getVarKindOffset(VarKind::Range));
  if (!rowBound.has_value()) {
    return failure();
  }
  auto colBound = expandedFilterRelation.getConstantBound64(
      BoundType::UB,
      expandedFilterRelation.getVarKindOffset(VarKind::Range) + 1);
  if (!colBound.has_value()) {
    return failure();
  }
  RankedTensorType expandedFilterType = RankedTensorType::get(
      {rowBound.value(), colBound.value()}, filterType.getElementType());

  auto diagonalizedFilterRelation =
      getDiagonalLayoutRelation(expandedFilterType, ciphertextSize);

  // Compose these relations.
  expandedFilterRelation.compose(diagonalizedFilterRelation);
  return expandedFilterRelation;
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

bool isRelation2dConvFilterDiagonalized(RankedTensorType filterType,
                                        RankedTensorType dataType,
                                        int64_t padding, int64_t ciphertextSize,
                                        presburger::IntegerRelation relation) {
  auto diagonalizedRelation = get2dConvFilterDiagonalizedRelation(
      filterType, dataType, padding, ciphertextSize);
  if (failed(diagonalizedRelation)) {
    return false;
  }
  return relation.isEqual(diagonalizedRelation.value());
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
  return *clonedRelation;
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
  return *clonedRelation;
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
  return *rel;
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

isl_stat pointPairCallback(__isl_take isl_point* pnt, void* user) {
  PointPairCollector* collector = static_cast<PointPairCollector*>(user);

  isl_space* space = isl_point_get_space(pnt);
  isl_space_free(space);

  std::vector<int64_t> domainPoint(collector->domainDims);
  std::vector<int64_t> rangePoint(collector->rangeDims);

  // Extract domain coordinates
  for (int i = 0; i < collector->domainDims; i++) {
    isl_val* coord = isl_point_get_coordinate_val(pnt, isl_dim_set, i);
    if (isl_val_is_int(coord)) {
      domainPoint[i] = isl_val_get_num_si(coord);
    }
    isl_val_free(coord);
  }

  // Extract range coordinates
  for (int i = 0; i < collector->rangeDims; i++) {
    isl_val* coord = isl_point_get_coordinate_val(pnt, isl_dim_set,
                                                  collector->domainDims + i);
    if (isl_val_is_int(coord)) {
      rangePoint[i] = isl_val_get_num_si(coord);
    }
    isl_val_free(coord);
  }

  collector->points.emplace_back(domainPoint, rangePoint);
  isl_point_free(pnt);
  return isl_stat_ok;
}

void enumeratePoints(const presburger::IntegerRelation& relation,
                     PointPairCollector& collector) {
  auto* bmap = convertRelationToBasicMap(relation, collector.ctx);
  isl_set* set = isl_set_from_basic_set(isl_basic_map_wrap(bmap));
  isl_set_foreach_point(set, &pointPairCallback, &collector);
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
