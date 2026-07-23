#include "lib/Utils/Layout/Utils.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

// ISL
#include "include/isl/ctx.h"         // from @isl
#include "include/isl/map.h"         // from @isl
#include "include/isl/map_type.h"    // from @isl
#include "include/isl/point.h"       // from @isl
#include "include/isl/set.h"         // from @isl
#include "include/isl/space.h"       // from @isl
#include "include/isl/space_type.h"  // from @isl
#include "include/isl/val.h"         // from @isl
#include "include/isl/val_type.h"    // from @isl

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

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
               std::optional<int64_t> upper) {
  result.addBound(BoundType::LB, pos, lower);
  if (upper.has_value()) {
    result.addBound(BoundType::UB, pos, upper.value());
  }
}

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

bool sameRangeForDomainPoint(const std::vector<int64_t>& domainPoint,
                             const presburger::IntegerRelation& rel1,
                             const presburger::IntegerRelation& rel2) {
  IntegerRelation fixedRel1 = fixDomainVars(rel1, domainPoint);
  IntegerRelation fixedRel2 = fixDomainVars(rel2, domainPoint);

  if (fixedRel1.computeVolume() != fixedRel1.computeVolume()) return false;

  // If this is still too slow, would it be faster to sample or enumerate range
  // points?
  return fixedRel1.isEqual(fixedRel2);
}

bool sameDomainForRangePoint(const std::vector<int64_t>& rangePoint,
                             const presburger::IntegerRelation& rel1,
                             const presburger::IntegerRelation& rel2) {
  IntegerRelation fixedRel1 = fixRangeVars(rel1, rangePoint);
  IntegerRelation fixedRel2 = fixRangeVars(rel2, rangePoint);

  if (fixedRel1.computeVolume() != fixedRel1.computeVolume()) return false;

  // If this is still too slow, would it be faster to sample or enumerate range
  // points?
  return fixedRel1.isEqual(fixedRel2);
}

LogicalResult tryProveUnequal(const presburger::IntegerRelation& layout1,
                              const presburger::IntegerRelation& layout2) {
  int64_t numDomain = layout1.getNumDomainVars();
  int64_t numRange = layout1.getNumRangeVars();

  if (numDomain != layout2.getNumDomainVars()) {
    return success();
  }
  if (numRange != layout2.getNumRangeVars()) {
    return success();
  }

  if (layout1.computeVolume() != layout2.computeVolume()) {
    return success();
  }

  std::vector<int64_t> domainVarBounds;
  for (int i = layout1.getVarKindOffset(VarKind::Domain);
       i < layout1.getVarKindEnd(VarKind::Domain); ++i) {
    auto layout1Bound = layout1.getConstantBound64(BoundType::UB, i);
    auto layout2Bound = layout2.getConstantBound64(BoundType::UB, i);
    if (layout1Bound != layout2Bound) {
      return success();
    }

    if (!layout1Bound.has_value() || !layout2Bound.has_value()) {
      return failure();
    }

    domainVarBounds.push_back(*layout1Bound);
  }

  std::vector<int64_t> rangeVarBounds;
  for (int i = layout1.getVarKindOffset(VarKind::Range);
       i < layout1.getVarKindEnd(VarKind::Range); ++i) {
    auto layout1Bound = layout1.getConstantBound64(BoundType::UB, i);
    auto layout2Bound = layout2.getConstantBound64(BoundType::UB, i);
    if (layout1Bound != layout2Bound) {
      return success();
    }

    if (!layout1Bound.has_value() || !layout2Bound.has_value()) {
      return failure();
    }

    rangeVarBounds.push_back(*layout1Bound);
  }

  // Since these are layouts mapping data tensors to ciphertext-semantic
  // tensors, both the domain and range spaces are simple grids from (0, 0, ...,
  // 0) to (bound0, bound1, ..., boundK). We can sample this grid however we
  // like, but it should suffice for most cases to check some corners and a few
  // interior points.

  std::vector<std::vector<int64_t>> domainPointsToTest;
  std::vector<int64_t> zeroDomain(0, numDomain);
  // a point on the diagonal, 1/3 along
  std::vector<int64_t> domainInterior1(0, numDomain);
  // a point on an anti-diagonal, 1/3 along
  std::vector<int64_t> domainInterior2(0, numDomain);
  for (int i = 0; i < numDomain; ++i) {
    domainInterior1.push_back(domainVarBounds[i] / 3);
    domainInterior2.push_back(i < numDomain / 2 ? domainVarBounds[i] / 3
                                                : 2 * domainVarBounds[i] / 3);
  }
  domainPointsToTest.push_back(std::move(zeroDomain));
  domainPointsToTest.push_back(domainVarBounds);
  domainPointsToTest.push_back(domainInterior1);
  domainPointsToTest.push_back(domainInterior2);

  std::vector<std::vector<int64_t>> rangePointsToTest;
  std::vector<int64_t> zeroRange(0, numRange);
  // a point on the diagonal, 1/3 along
  std::vector<int64_t> rangeInterior1(0, numRange);
  // a point on an anti-diagonal, 1/3 along
  std::vector<int64_t> rangeInterior2(0, numRange);
  for (int i = 0; i < numRange; ++i) {
    rangeInterior1.push_back(rangeVarBounds[i] / 3);
    rangeInterior2.push_back(i < numRange / 2 ? rangeVarBounds[i] / 3
                                              : 2 * rangeVarBounds[i] / 3);
  }
  rangePointsToTest.push_back(std::move(zeroRange));
  rangePointsToTest.push_back(rangeVarBounds);
  rangePointsToTest.push_back(rangeInterior1);
  rangePointsToTest.push_back(rangeInterior2);

  for (const auto& domainPt : domainPointsToTest) {
    if (!sameRangeForDomainPoint(domainPt, layout1, layout2)) {
      return success();
    }
  }

  for (const auto& rangePt : rangePointsToTest) {
    if (!sameDomainForRangePoint(rangePt, layout1, layout2)) {
      return success();
    }
  }

  return failure();
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
  // matrix, so ensure that the diagonal size is less than
  // the result's columns.
  assert(std::max(rows, cols) <= ciphertextSize);

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
  int64_t numDiagonals = std::min(paddedRows, paddedCols);
  result.addBound(BoundType::UB, rangeOffset, numDiagonals - 1);
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

FailureOr<presburger::IntegerRelation> diagonalize2dMatrix(
    presburger::IntegerRelation relation, RankedTensorType originalType,
    int64_t ciphertextSize) {
  // Get size of the matrix.
  auto rowBound = relation.getConstantBound64(
      BoundType::UB, relation.getVarKindOffset(VarKind::Range));
  auto colBound = relation.getConstantBound64(
      BoundType::UB, relation.getVarKindOffset(VarKind::Range) + 1);
  if (!rowBound.has_value() || !colBound.has_value()) {
    return failure();
  }
  RankedTensorType matrixType =
      RankedTensorType::get({rowBound.value() + 1, colBound.value() + 1},
                            originalType.getElementType());
  auto diagonalRelation = getDiagonalLayoutRelation(matrixType, ciphertextSize);

  // Compose these relations.
  relation.compose(diagonalRelation);
  return relation;
}

presburger::IntegerRelation getBicyclicLayoutRelation(
    RankedTensorType matrixType, int64_t numSlots) {
  unsigned int rows = matrixType.getDimSize(0);
  unsigned int cols = matrixType.getDimSize(1);

  assert(std::gcd(rows, cols) == 1 &&
         "bicyclic layout requires coprime dimensions");

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      matrixType.getRank(), /*numRange=*/2, /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for the data matrix dimensions.
  int domainOffset = result.getVarKindOffset(VarKind::Domain);
  int rangeOffset = result.getVarKindOffset(VarKind::Range);
  int rowVarIndex = domainOffset;
  int colVarIndex = domainOffset + 1;
  int ctVarIndex = rangeOffset;
  int slotVarIndex = rangeOffset + 1;

  addBounds(result, rowVarIndex, 0, rows - 1);
  addBounds(result, colVarIndex, 0, cols - 1);
  addBounds(result, ctVarIndex, 0,
            std::ceil((float)matrixType.getNumElements() / numSlots) - 1);
  addBounds(result, slotVarIndex, 0, numSlots - 1);

  // Let k = ct * numSlots + slot.
  // We need to add constraints for:
  // row = k % rows
  // col = k % cols

  // k_mod_rows = (ct * numSlots + slot) % rows
  SmallVector<int64_t> kCoeffs(result.getNumCols(), 0);
  kCoeffs[ctVarIndex] = numSlots;
  kCoeffs[slotVarIndex] = 1;
  auto kModRows = addModConstraint(result, kCoeffs, rows);

  // row = k_mod_rows
  SmallVector<int64_t> rowEquality(result.getNumCols(), 0);
  rowEquality[rowVarIndex] = 1;
  rowEquality[kModRows] = -1;
  result.addEquality(rowEquality);

  // k_mod_cols = (ct * numSlots + slot) % cols
  kCoeffs.resize(result.getNumCols(), 0);
  kCoeffs[ctVarIndex] = numSlots;
  kCoeffs[slotVarIndex] = 1;
  auto kModCols = addModConstraint(result, kCoeffs, cols);

  // col = k_mod_cols
  SmallVector<int64_t> colEquality(result.getNumCols(), 0);
  colEquality[colVarIndex] = 1;
  colEquality[kModCols] = -1;
  result.addEquality(colEquality);

  return result;
}

// Returns an IntegerRelation representing the tricyclic encoding mapping for
// a 3-D tensor of shape (h, m, n) into ciphertext slots. The relation maps
// domain vars [h_idx, m_idx, n_idx] to range vars [ct, slot] via
// k = ct * numSlots + slot and constraining:
//   h_idx == k % h
//   m_idx == k % m
//   n_idx == k % n
presburger::IntegerRelation getTricyclicLayoutRelation(
    RankedTensorType tensorType, int64_t numSlots) {
  assert(tensorType.getRank() == 3 && "tricyclic layout expects a 3-D tensor");

  int64_t h = tensorType.getDimSize(0);
  int64_t m = tensorType.getDimSize(1);
  int64_t n = tensorType.getDimSize(2);

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      tensorType.getRank(), /*numRange=*/2, /*numSymbol=*/0,
      /*numLocals=*/0));

  // Setup var indices
  int domainOffset = result.getVarKindOffset(VarKind::Domain);
  int rangeOffset = result.getVarKindOffset(VarKind::Range);
  int hVarIndex = domainOffset;
  int mVarIndex = domainOffset + 1;
  int nVarIndex = domainOffset + 2;
  int ctVarIndex = rangeOffset;
  int slotVarIndex = rangeOffset + 1;

  // Add bounds for domain and range variables.
  addBounds(result, hVarIndex, 0, h - 1);
  addBounds(result, mVarIndex, 0, m - 1);
  addBounds(result, nVarIndex, 0, n - 1);
  addBounds(result, ctVarIndex, 0,
            std::ceil((float)tensorType.getNumElements() / numSlots) - 1);
  addBounds(result, slotVarIndex, 0, numSlots - 1);

  // Let k = ct * numSlots + slot.
  // We need constraints:
  //   h_idx = k % h
  //   m_idx = k % m
  //   n_idx = k % n

  // k_mod_h = (ct * numSlots + slot) % h
  SmallVector<int64_t> kCoeffs(result.getNumCols(), 0);
  kCoeffs[ctVarIndex] = numSlots;
  kCoeffs[slotVarIndex] = 1;
  auto kModH = addModConstraint(result, kCoeffs, h);

  // h_idx = k_mod_h
  SmallVector<int64_t> hEquality(result.getNumCols(), 0);
  hEquality[hVarIndex] = 1;
  hEquality[kModH] = -1;
  result.addEquality(hEquality);

  // k_mod_m = (ct * numSlots + slot) % m
  kCoeffs.assign(result.getNumCols(), 0);
  kCoeffs[ctVarIndex] = numSlots;
  kCoeffs[slotVarIndex] = 1;
  auto kModM = addModConstraint(result, kCoeffs, m);

  // m_idx = k_mod_m
  SmallVector<int64_t> mEquality(result.getNumCols(), 0);
  mEquality[mVarIndex] = 1;
  mEquality[kModM] = -1;
  result.addEquality(mEquality);

  // k_mod_n = (ct * numSlots + slot) % n
  kCoeffs.assign(result.getNumCols(), 0);
  kCoeffs[ctVarIndex] = numSlots;
  kCoeffs[slotVarIndex] = 1;
  auto kModN = addModConstraint(result, kCoeffs, n);

  // n_idx = k_mod_n
  SmallVector<int64_t> nEquality(result.getNumCols(), 0);
  nEquality[nVarIndex] = 1;
  nEquality[kModN] = -1;
  result.addEquality(nEquality);

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

bool isOneToOneSingleCiphertextPacking(
    const presburger::IntegerRelation& relation) {
  if (relation.getNumDomainVars() != 1 || relation.getNumRangeVars() != 2)
    return false;

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* map = convertRelationToBasicMap(relation, ctx);
  if (!map) {
    isl_ctx_free(ctx);
    return false;
  }

  isl_val* ct =
      isl_basic_map_plain_get_val_if_fixed(map, isl_dim_out, /*pos=*/0);
  bool singleCiphertext = ct && isl_val_is_zero(ct) == isl_bool_true;
  isl_val_free(ct);

  isl_basic_map* slots = isl_basic_map_project_out(
      isl_basic_map_copy(map), isl_dim_out, /*first=*/0, /*n=*/1);
  isl_map* slotMap = isl_map_from_basic_map(slots);
  isl_bool oneToOne = isl_map_is_bijective(slotMap);
  isl_map_free(slotMap);
  isl_basic_map_free(map);
  isl_ctx_free(ctx);
  return singleCiphertext && oneToOne == isl_bool_true;
}

IntegerRelation foldVectorPermutationIntoMatrixLayout(
    const IntegerRelation& vectorPermutation,
    const IntegerRelation& matrixLayout) {
  // vectorPermutation maps a vector index [col] -> [ct, slot] as a
  // single-ciphertext permutation (ct is fixed to zero). Drop the constant ct
  // output, leaving the pure index-to-slot permutation [col] -> [slot].
  IntegerRelation result(vectorPermutation);
  result.projectOut(result.getVarKindOffset(VarKind::Range), 1);

  // Lift the permutation to a matrix domain by prepending a passthrough row
  // dimension to both sides (row_in == row_out), giving
  // [row, col] -> [row, slot].
  result.insertVar(VarKind::Domain, 0, 1);
  result.insertVar(VarKind::Range, 0, 1);
  SmallVector<int64_t> rowEq(result.getNumCols(), 0);
  rowEq[result.getVarKindOffset(VarKind::Domain)] = 1;
  rowEq[result.getVarKindOffset(VarKind::Range)] = -1;
  result.addEquality(rowEq);

  // Compose with the matrix layout (result;matrixLayout): the permutation's
  // [row, slot] output feeds the matrix layout's [row, col] input, so the
  // vector permutation is absorbed into the matrix's column indexing, yielding
  // the folded matrix layout [row, col] -> [ct, slot].
  result.compose(matrixLayout);
  result.removeRedundantConstraints();
  result.simplify();
  return result;
}

bool isRelationPerRow(RankedTensorType matrixType, int64_t ciphertextSize,
                      presburger::IntegerRelation relation) {
  IntegerRelation perRowRelation =
      getPerRowLayoutRelation(matrixType, ciphertextSize);
  return relation.isEqual(perRowRelation);
}

bool isRelationBicyclic(RankedTensorType matrixType, int64_t numSlots,
                        const presburger::IntegerRelation& relation) {
  // Reject non-co-prime dimensions.
  if (matrixType.getRank() != 2) return false;
  unsigned int rows = matrixType.getDimSize(0);
  unsigned int cols = matrixType.getDimSize(1);
  if (std::gcd(rows, cols) != 1) return false;
  IntegerRelation bicyclicRelation =
      getBicyclicLayoutRelation(matrixType, numSlots);
  return relation.isEqual(bicyclicRelation);
}

bool isRelationTricyclic(RankedTensorType tensorType, int64_t numSlots,
                         const presburger::IntegerRelation& relation) {
  // Reject non-co-prime dimensions.
  if (tensorType.getRank() != 3) return false;
  int64_t h = tensorType.getDimSize(0);
  int64_t m = tensorType.getDimSize(1);
  int64_t n = tensorType.getDimSize(2);
  if (std::gcd(h, m) != 1 || std::gcd(m, n) != 1 || std::gcd(h, n) != 1)
    return false;
  IntegerRelation tricyclicRelation =
      getTricyclicLayoutRelation(tensorType, numSlots);
  return relation.isEqual(tricyclicRelation);
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
    // Iterate starting from the largest index so that earlier deletion do not
    // impact later indices
    for (int64_t reassocDim : llvm::reverse(associationGroup)) {
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

  // Handle the case where reassociation is empty
  if (reassociation.empty()) {
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      auto newDimIndex = clonedRelation->insertVar(VarKind::Domain, i);
      clonedRelation->addBound(BoundType::LB, newDimIndex, 0);
      clonedRelation->addBound(BoundType::UB, newDimIndex, 0);
    }
    return *clonedRelation;
  }

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
  assert(static_cast<int64_t>(clonedRelation->getNumDomainVars()) ==
             resultType.getRank() &&
         "expandDimensions: result relation domain rank must match the result "
         "tensor rank");
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

namespace {
// Extract the first `n` set coordinates of `pnt` into `out`.
void extractCoords(__isl_keep isl_point* pnt, int n,
                   std::vector<int64_t>& out) {
  for (int i = 0; i < n; i++) {
    isl_val* coord = isl_point_get_coordinate_val(pnt, isl_dim_set, i);
    if (isl_val_is_int(coord)) {
      out[i] = isl_val_get_num_si(coord);
    }
    isl_val_free(coord);
  }
}

// Computes each domain variable's box bounds [lb, ub].
//
// It first tries a quick scan for single-variable equality/inequality rows of
// `rel` (e.g. `c = 0`, `f >= 0`, `7 - f >= 0`), which the layout construction
// adds for the data-tensor index space. This avoids
// IntegerRelation::getConstantBound64, which derives bounds via Fourier-Motzkin
// elimination over every other variable and blows up on the relation's
// mod/floordiv existentials. For any bound the quick scan cannot determine, it
// falls back to getConstantBound64 for that single bound.
void getDomainBox(const presburger::IntegerRelation& rel,
                  SmallVector<int64_t>& lb, SmallVector<int64_t>& ub) {
  auto floorDivI = [](int64_t a, int64_t b) -> int64_t {
    int64_t q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) --q;
    return q;
  };
  unsigned numDomain = rel.getNumDomainVars();
  unsigned numVars = rel.getNumVars();
  unsigned constCol = rel.getNumCols() - 1;
  unsigned numIneqs = rel.getNumInequalities();
  lb.assign(numDomain, std::numeric_limits<int64_t>::min());
  ub.assign(numDomain, std::numeric_limits<int64_t>::max());

  // Single pass over every constraint (inequalities then equalities, per
  // atConstraint64's indexing). A constraint bounds a domain variable only if
  // it has exactly one nonzero variable coefficient and that variable is a
  // domain variable. Inequalities (coeff*v + c >= 0) give one bound by sign;
  // equalities (coeff*v + c = 0) pin both bounds.
  for (unsigned r = 0, e = rel.getNumConstraints(); r < e; ++r) {
    int soleVar = -1;
    bool multiple = false;
    for (unsigned j = 0; j < numVars; ++j) {
      if (rel.atConstraint64(r, j) == 0) continue;
      if (soleVar != -1) {
        multiple = true;
        break;
      }
      soleVar = static_cast<int>(j);
    }
    if (multiple || soleVar < 0 || static_cast<unsigned>(soleVar) >= numDomain)
      continue;

    unsigned v = soleVar;
    int64_t coeff = rel.atConstraint64(r, v);
    int64_t c = rel.atConstraint64(r, constCol);
    if (r < numIneqs) {
      if (coeff > 0) {
        lb[v] = std::max(lb[v], -floorDivI(c, coeff));  // v >= ceil(-c/coeff)
      } else {
        ub[v] = std::min(ub[v], floorDivI(c, -coeff));  // v <= floor(c/-coeff)
      }
    } else {
      // A single-variable equality coeff*v + c = 0 has an integer solution
      // only if coeff divides c; otherwise the relation is infeasible.
      assert((-c) % coeff == 0 && "non-integer single-variable equality bound");
      int64_t val = -c / coeff;
      lb[v] = std::max(lb[v], val);
      ub[v] = std::min(ub[v], val);
    }
  }

  // Fall back to the general (but expensive) constant-bound computation for any
  // domain variable the quick single-variable scan left undetermined. This uses
  // Fourier-Motzkin elimination over the other variables, so we only reach it
  // when the cheap method is insufficient.
  for (unsigned i = 0; i < numDomain; ++i) {
    if (lb[i] == std::numeric_limits<int64_t>::min()) {
      std::optional<int64_t> b = rel.getConstantBound64(BoundType::LB, i);
      if (!b)
        llvm::report_fatal_error(
            "getDomainBox: domain variable has no constant lower bound");
      lb[i] = *b;
    }
    if (ub[i] == std::numeric_limits<int64_t>::max()) {
      std::optional<int64_t> b = rel.getConstantBound64(BoundType::UB, i);
      if (!b)
        llvm::report_fatal_error(
            "getDomainBox: domain variable has no constant upper bound");
      ub[i] = *b;
    }
  }
}

// Enumerates the domain box [lb, ub] explicitly and, for each concrete domain
// point, invokes `onImagePoint(domainPoint, imagePoint)` for every point of
// that domain point's image. `imagePoint` is borrowed (freed by this function).
// Between domain points `shouldStop` is polled (when non-null); returning true
// halts enumeration early.
//
// Fixing the domain to concrete integers collapses the relation's mod/floordiv
// existentials into a non-parametric feasibility problem that isl resolves
// cheaply. We deliberately do NOT ask isl to enumerate the domain set
// (isl_set_foreach_point over isl_basic_map_domain): projecting the range out
// leaves those existentials in the domain set, making that scan itself an
// expensive parametric-ILP solve -- the very cost we are avoiding.
void forEachDomainImagePoint(
    __isl_keep isl_basic_map* bmap, isl_ctx* ctx, ArrayRef<int64_t> lb,
    ArrayRef<int64_t> ub,
    llvm::function_ref<void(ArrayRef<int64_t>, __isl_keep isl_point*)>
        onImagePoint,
    llvm::function_ref<bool()> shouldStop = nullptr) {
  unsigned numDomain = lb.size();

  // isl_set_foreach_point takes a C callback; bridge to onImagePoint through
  // this context and free the (owned) point afterwards.
  struct ImageCtx {
    llvm::function_ref<void(ArrayRef<int64_t>, isl_point*)> cb;
    ArrayRef<int64_t> domainPoint;
  };
  auto trampoline = [](__isl_take isl_point* pnt, void* user) -> isl_stat {
    auto* ic = static_cast<ImageCtx*>(user);
    ic->cb(ic->domainPoint, pnt);
    isl_point_free(pnt);
    return isl_stat_ok;
  };

  auto imageOf = [&](ArrayRef<int64_t> point) {
    isl_basic_map* fixed = isl_basic_map_copy(bmap);
    for (unsigned i = 0; i < numDomain; ++i) {
      fixed = isl_basic_map_fix_val(fixed, isl_dim_in, i,
                                    isl_val_int_from_si(ctx, point[i]));
    }
    isl_set* image = isl_set_from_basic_set(isl_basic_map_range(fixed));
    ImageCtx ic{onImagePoint, point};
    isl_set_foreach_point(image, trampoline, &ic);
    isl_set_free(image);
  };

  if (numDomain == 0) {
    imageOf({});
    return;
  }
  // An empty box (some lb > ub) contains no points.
  for (unsigned i = 0; i < numDomain; ++i)
    if (lb[i] > ub[i]) return;

  SmallVector<int64_t> point(lb.begin(), lb.end());
  while (true) {
    imageOf(point);
    if (shouldStop && shouldStop()) break;
    // Advance the mixed-radix odometer over the domain box.
    int d = static_cast<int>(numDomain) - 1;
    for (; d >= 0; --d) {
      if (++point[d] <= ub[d]) break;
      point[d] = lb[d];
    }
    if (d < 0) break;
  }
}
}  // namespace

void enumeratePoints(const presburger::IntegerRelation& relation,
                     PointPairCollector& collector) {
  assert(relation.getNumDomainVars() ==
             static_cast<unsigned>(collector.domainDims) &&
         "collector domainDims must match the relation's domain rank");
  isl_basic_map* bmap = convertRelationToBasicMap(relation, collector.ctx);

  SmallVector<int64_t> lb, ub;
  getDomainBox(relation, lb, ub);

  forEachDomainImagePoint(
      bmap, collector.ctx, lb, ub,
      [&](ArrayRef<int64_t> domainPoint, __isl_keep isl_point* imagePoint) {
        std::vector<int64_t> rangePoint(collector.rangeDims);
        extractCoords(imagePoint, collector.rangeDims, rangePoint);
        collector.points.emplace_back(
            std::vector<int64_t>(domainPoint.begin(), domainPoint.end()),
            std::move(rangePoint));
      });

  isl_basic_map_free(bmap);
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

void getCtComplementPoints(const presburger::IntegerRelation& relation,
                           PointCollector& collector,
                           RankedTensorType outputType) {
  // Assert precondition that there must be two range vars.
  assert(relation.getNumRangeVars() == 2 && "Expected 2 range vars.");
  assert(outputType.getRank() == 2 && "Expected 2D output type.");

  int64_t numCts = outputType.getDimSize(0);

  SmallVector<int64_t> lb, ub;
  getDomainBox(relation, lb, ub);

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);

  // Mark which ct indices (range var 0) actually appear in the range. See
  // forEachDomainImagePoint for why we enumerate the domain rather than probe
  // each ct with isl_basic_map_is_empty. Once every ct is accounted for we can
  // stop early instead of scanning the rest of a large domain.
  std::vector<bool> seen(numCts, false);
  int64_t seenCount = 0;
  forEachDomainImagePoint(
      bmap, ctx, lb, ub,
      [&](ArrayRef<int64_t> /*domainPoint*/, __isl_keep isl_point* imagePoint) {
        isl_val* coord =
            isl_point_get_coordinate_val(imagePoint, isl_dim_set, 0);
        if (isl_val_is_int(coord)) {
          int64_t ct = isl_val_get_num_si(coord);
          if (ct >= 0 && ct < numCts && !seen[ct]) {
            seen[ct] = true;
            ++seenCount;
          }
        }
        isl_val_free(coord);
      },
      /*shouldStop=*/[&]() { return seenCount == numCts; });

  isl_basic_map_free(bmap);
  isl_ctx_free(ctx);

  // The complement is every ct index in [0, numCts) that never appeared,
  // emitted in ascending order.
  for (int64_t ct = 0; ct < numCts; ++ct) {
    if (!seen[ct]) {
      collector.points.push_back({ct});
    }
  }
}

presburger::IntegerRelation getCollapsedRelation(
    RankedTensorType sourceType, RankedTensorType destType,
    ArrayRef<ReassociationIndices> reassociation) {
  auto domainSize = sourceType.getRank();
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      domainSize, /*numRange=*/reassociation.size(), /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for the source dimensions.
  for (int i = 0; i < sourceType.getRank(); ++i) {
    addBounds(result, i, 0, sourceType.getDimSize(i) - 1);
  }

  auto domainOffset = result.getVarKindOffset(VarKind::Domain);
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  for (auto [idx, group] : llvm::enumerate(reassociation)) {
    // Add bounds for the collapsed dimension.
    addBounds(result, rangeOffset + idx, 0, destType.getDimSize(idx) - 1);
    // Add an equality constraint that takes a row major relation of each
    // group of source indices and set that equal to the idx'th range
    // variable.
    SmallVector<int64_t> rowMajorCoeffs(result.getNumCols(), 0);
    unsigned product = 1;
    for (int64_t dim : llvm::reverse(group)) {
      rowMajorCoeffs[domainOffset + dim] = product;
      product *= sourceType.getDimSize(dim);
    }
    rowMajorCoeffs[rangeOffset + idx] = -1;
    result.addEquality(rowMajorCoeffs);
  }

  return result;
}

FailureOr<presburger::IntegerRelation> getSliceInsertionRelation(
    RankedTensorType sliceType, RankedTensorType resultType,
    SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
    SmallVector<int64_t> strides) {
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      sliceType.getRank(), /*numRange=*/resultType.getRank(), /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for the source dimensions.
  auto domainOffset = result.getVarKindOffset(VarKind::Domain);
  for (int i = 0; i < sliceType.getRank(); ++i) {
    auto dimSize = sliceType.getDimSize(i);
    if (!ShapedType::isDynamic(dimSize)) {
      addBounds(result, domainOffset + i, 0, dimSize - 1);
    }
  }

  // Add bounds for the result dimensions.
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  for (int i = 0; i < resultType.getRank(); ++i) {
    auto dimSize = resultType.getDimSize(i);
    if (!ShapedType::isDynamic(dimSize)) {
      addBounds(result, rangeOffset + i, 0, dimSize - 1);
    }
  }

  // Source tensor's dimensions (d0, d1, ...) are mapped sequentially to the
  // destination tensor's dimensions (r0, r1, ...) for which the slice size is
  // greater than 1.
  auto constOffset = result.getNumCols() - 1;
  unsigned int sourceDim = 0;
  for (auto destDim = 0; destDim < resultType.getRank(); ++destDim) {
    if (sizes[destDim] > 1) {
      // Map from the i-th source dimension
      // r_j = offsets[j] + d_i * strides[j]
      addConstraint(result,
                    {{rangeOffset + destDim, -1},
                     {constOffset, offsets[destDim]},
                     {domainOffset + sourceDim, strides[destDim]}},
                    /*equality=*/true);
      ++sourceDim;
    } else {
      // This is a dropped dimension, fixed at the offset
      // r_j = offsets[j]
      addConstraint(
          result,
          {{rangeOffset + destDim, -1}, {constOffset, offsets[destDim]}},
          /*equality=*/true);
    }
  }

  return result;
}

presburger::IntegerRelation shiftVar(
    const presburger::IntegerRelation& relation, unsigned int pos,
    int64_t offset) {
  auto varKind = relation.getVarKindAt(pos);
  auto varKindOffset = relation.getVarKindOffset(varKind);
  auto shiftedRelation = relation.clone();
  // Add a new var var' at pos, set var' = var + offset, and then eliminate
  // var at position pos + 1
  shiftedRelation->insertVar(varKind, pos - varKindOffset, 1);
  addConstraint(
      *shiftedRelation,
      {{pos, 1}, {pos + 1, -1}, {shiftedRelation->getNumCols() - 1, -offset}},
      /*equality=*/true);
  shiftedRelation->projectOut(pos + 1, 1);
  return *shiftedRelation;
}

FailureOr<presburger::IntegerRelation> getSliceExtractionRelation(
    RankedTensorType sourceType, RankedTensorType resultType,
    SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
    SmallVector<int64_t> strides) {
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      sourceType.getRank(), /*numRange=*/resultType.getRank(),
      /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for the source dimensions.
  auto domainOffset = result.getVarKindOffset(VarKind::Domain);
  for (int i = 0; i < sourceType.getRank(); ++i) {
    addBounds(result, domainOffset + i, 0, sourceType.getDimSize(i) - 1);
  }

  // Add bounds for the result dimensions.
  auto rangeOffset = result.getVarKindOffset(VarKind::Range);
  for (int i = 0; i < resultType.getRank(); ++i) {
    addBounds(result, rangeOffset + i, 0, resultType.getDimSize(i) - 1);
  }

  // Destination tensor's dimensions (d0, d1, ...) are mapped sequentially
  // from the source tensor's dimensions (r0, r1, ...) for which the slice
  // size is greater than 1.
  auto constOffset = result.getNumCols() - 1;
  unsigned int resultDim = 0;
  for (auto sourceDim = 0; sourceDim < sourceType.getRank(); ++sourceDim) {
    if (sizes[sourceDim] > 1) {
      // Map to the i-th result dimension
      // d_j = offsets[j] + r_i * strides[j]
      addConstraint(result,
                    {{domainOffset + sourceDim, -1},
                     {constOffset, offsets[sourceDim]},
                     {rangeOffset + resultDim, strides[sourceDim]}},
                    /*equality=*/true);
      ++resultDim;
    } else {
      // This is a dropped dimension, fixed at the offset
      // d_j = offsets[j]
      addConstraint(
          result,
          {{domainOffset + sourceDim, -1}, {constOffset, offsets[sourceDim]}},
          /*equality=*/true);
    }
  }

  return result;
}

bool isRelationEqual(const presburger::IntegerRelation& relation1,
                     const presburger::IntegerRelation& relation2) {
  bool fastCheck = relation1.isObviouslyEqual(relation2);
  if (fastCheck) return true;

  LogicalResult inequalityTest = tryProveUnequal(relation2, relation1);
  if (succeeded(inequalityTest)) return false;

  bool slowCheck = relation1.isEqual(relation2);
  return slowCheck;
}

bool isDenseLayout(const presburger::IntegerRelation& relation,
                   RankedTensorType type) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);

  if (!bmap) {
    isl_ctx_free(ctx);
    return false;
  }

  // Get the range set from the basic_map of the relation
  isl_basic_set* rangeBset = isl_basic_map_range(bmap);
  isl_set* imageSet = isl_set_from_basic_set(rangeBset);

  // The number of type dimensions should match the range variables
  unsigned numDims = type.getRank();
  if (isl_set_dim(imageSet, isl_dim_set) != numDims) {
    isl_set_free(imageSet);
    isl_ctx_free(ctx);
    return false;
  }

  // Construct a full dense set for the given shape
  isl_space* space = isl_set_get_space(imageSet);
  isl_set* denseSet = isl_set_universe(space);
  for (unsigned i = 0; i < numDims; ++i) {
    // 0 <= d_i < dim[i]
    denseSet = isl_set_lower_bound_val(denseSet, isl_dim_set, i,
                                       isl_val_int_from_si(ctx, 0));
    denseSet = isl_set_upper_bound_val(
        denseSet, isl_dim_set, i,
        isl_val_int_from_si(ctx, type.getDimSize(i) - 1));
  }

  bool result = isl_set_is_equal(imageSet, denseSet) == isl_bool_true;

  isl_set_free(imageSet);
  isl_set_free(denseSet);
  isl_ctx_free(ctx);

  return result;
}

int64_t relationSize(const IntegerRelation& rel) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(rel, ctx);
  isl_set* set = isl_set_from_basic_set(isl_basic_map_wrap(bmap));

  if (isl_set_is_bounded(set) != isl_bool_true) {
    isl_set_free(set);
    isl_ctx_free(ctx);
    return -1;
  }

  isl_val* card = isl_set_count_val(set);

  if (!card || isl_val_is_nan(card)) {
    if (card) isl_val_free(card);
    isl_set_free(set);
    isl_ctx_free(ctx);
    return -1;
  }

  int64_t count = -1;
  if (isl_val_is_int(card) && isl_val_cmp_si(card, LONG_MAX) <= 0) {
    count = isl_val_get_num_si(card);
  }
  isl_val_free(card);
  isl_set_free(set);
  isl_ctx_free(ctx);
  return count;
}

}  // namespace heir
}  // namespace mlir
