#ifndef LIB_UTILS_LAYOUT_UTILS_H_
#define LIB_UTILS_LAYOUT_UTILS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

// ISL
#include "include/isl/ctx.h"  // from @isl
#include "include/isl/map.h"  // from @isl

namespace mlir {
namespace heir {

// Helper that adds constraints built from the array of positions and coeffs.
// Inequalities are given by (>= 0).
void addConstraint(presburger::IntegerRelation& result,
                   ArrayRef<std::pair<int64_t, int64_t>> posAndCoeff,
                   bool equality);
// Helper that adds inclusive lower and upper bounds for a given position and
// value.
void addBounds(presburger::IntegerRelation& result, int64_t pos, int64_t lower,
               std::optional<int64_t> upper = std::nullopt);

// Adds a new local variable q to the relation that represents expr % modulus.
// Returns the index of the new local variable in the relation.
unsigned int addModConstraint(presburger::IntegerRelation& result,
                              ArrayRef<int64_t> exprs, int64_t modulus);

// Returns an IntegerRelation that enforces a row-major layout for the given
// tensor type and number of slots. This is used for IntegerRelations that
// represent data layouts in ciphertexts. It expects that the number of domain
// variables match the rank of the tensor, and that there are two range
// variables representing the ciphertext index and slot index in that order.
presburger::IntegerRelation getRowMajorLayoutRelation(
    RankedTensorType tensorType, int64_t numSlots);

// Returns an IntegerRelation that represents a diagonalized layout for a matrix
// such that the ith diagonal of the matrix is in the ith row of the
// result. The number of rows of the input and output must match.
presburger::IntegerRelation getDiagonalLayoutRelation(
    RankedTensorType matrixType, int64_t ciphertextSize);

// Returns an IntegerRelation that represents a bicyclic layout for a matrix.
// See https://eprint.iacr.org/2024/1762 for details.
presburger::IntegerRelation getBicyclicLayoutRelation(
    RankedTensorType matrixType, int64_t numSlots);

// Returns an IntegerRelation that represents a per-row layout for a matrix
// such that each row of the matrix is in a separate ciphertext.
presburger::IntegerRelation getPerRowLayoutRelation(RankedTensorType matrixType,
                                                    int64_t ciphertextSize);

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication. This does not include diagonalizing
// the matrix, this simply returns the expanded data matrix.
// TODO(#2217): Support non-unit strides.
presburger::IntegerRelation get2dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    int64_t padding);

RankedTensorType get2dConvFilterExpandedType(RankedTensorType filterType,
                                             RankedTensorType dataType,
                                             int64_t padding);

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication.
// TODO(#2217): Support non-unit strides.
FailureOr<presburger::IntegerRelation> get2dConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize);

// Returns true if the given relation is a squat diagonal layout for the given
// matrix type and ciphertext semantic shape.
bool isRelationSquatDiagonal(RankedTensorType matrixType,
                             int64_t ciphertextSize,
                             const presburger::IntegerRelation& relation);

// Returns true if the given relation is a row-major layout for the given
// vector type and slot size.
bool isRelationRowMajor(RankedTensorType vectorType, int64_t numSlots,
                        const presburger::IntegerRelation& relation);

// Returns true if the given relation is a per-row layout
// for the given matrix type and ciphertext semantic shape.
bool isRelationPerRow(RankedTensorType matrixType, int64_t ciphertextSize,
                      presburger::IntegerRelation relation);

bool isRelation2dConvFilterDiagonalized(RankedTensorType filterType,
                                        RankedTensorType dataType,
                                        int64_t padding, int64_t ciphertextSize,
                                        presburger::IntegerRelation relation);

// Returns true if the given relation is a bicyclic layout for the given
// matrix type and ciphertext semantic shape.
bool isRelationBicyclic(RankedTensorType matrixType, int64_t numSlots,
                        const presburger::IntegerRelation& relation);

// Returns a new IntegerRelation that is the same as the given relation, but
// with the given dimensions collapsed. This expects that the reassociation
// indices result in a rank-reduction of the source type (i.e. the collapsed
// dimensions are all unit dimensions).
presburger::IntegerRelation collapseDimensions(
    const presburger::IntegerRelation& relation, RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociation);

// Returns a new IntegerRelation that is the same as the given relation, but
// with the given dimensions expanded. This expects that the reassociation
// indices result in a rank-expansion of the result type (i.e. the expanded
// dimensions are all unit dimensions).
presburger::IntegerRelation expandDimensions(
    const presburger::IntegerRelation& relation, RankedTensorType resultType,
    ArrayRef<ReassociationIndices> reassociation);

// Returns a new relation produced by constraining the index dimensions of
// type varKind to the given relation to the provided values. The fixedValues
// array size should equal the number of variables of type varKind.
presburger::IntegerRelation fixVars(const presburger::IntegerRelation& relation,
                                    ArrayRef<int64_t> fixedValues,
                                    presburger::VarKind varKind);

// Returns a new relation produced by constraining the domain variables of the
// given relation to the provided values.
//
// The fixedValues array should have size equal to the number of domain
// variables in the same order as `relation`. This generally should align with
// the order of the dimensions of the RankedTensorType this relation is laying
// out.
inline presburger::IntegerRelation fixDomainVars(
    const presburger::IntegerRelation& relation,
    ArrayRef<int64_t> fixedValues) {
  return fixVars(relation, fixedValues, presburger::VarKind::Domain);
}

inline presburger::IntegerRelation fixRangeVars(
    const presburger::IntegerRelation& relation,
    ArrayRef<int64_t> fixedValues) {
  return fixVars(relation, fixedValues, presburger::VarKind::Range);
}

struct PointCollector {
  std::vector<std::vector<int64_t>> points;
  isl_ctx* ctx;

  PointCollector() { ctx = isl_ctx_alloc(); }

  ~PointCollector() { isl_ctx_free(ctx); }

  // Delete copy constructor and assignment to avoid double-free
  PointCollector(const PointCollector&) = delete;
  PointCollector& operator=(const PointCollector&) = delete;
};

struct PointPairCollector {
  using Point = std::vector<int64_t>;
  std::vector<std::pair<Point, Point>> points;
  isl_ctx* ctx;
  int domainDims;
  int rangeDims;

  PointPairCollector(int domainDims, int rangeDims)
      : domainDims(domainDims), rangeDims(rangeDims) {
    ctx = isl_ctx_alloc();
  }

  ~PointPairCollector() { isl_ctx_free(ctx); }

  // Delete copy constructor and assignment to avoid double-free
  PointPairCollector(const PointPairCollector&) = delete;
  PointPairCollector& operator=(const PointPairCollector&) = delete;
};

// Get a list of points in the relation by enumerating all possible values.
void enumeratePoints(const presburger::IntegerRelation& relation,
                     PointPairCollector& collector);

// Get a list of points in the range of the relation by enumerating all
// possible values.
void getRangePoints(const presburger::IntegerRelation& relation,
                    PointCollector& collector);

// Sample a point in the range of the relation.
std::vector<int64_t> anyRangePoint(const presburger::IntegerRelation& relation);

// Collapse a relation with the given reassociation indices. Dimensions that are
// collapsed in a row-major order.
presburger::IntegerRelation getCollapsedRelation(
    RankedTensorType sourceType, RankedTensorType destType,
    ArrayRef<ReassociationIndices> reassociation);

// Get layout relation that corresponds to a tensor::insert_slice op.
FailureOr<presburger::IntegerRelation> getSliceInsertionRelation(
    RankedTensorType sliceType, RankedTensorType resultType,
    SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
    SmallVector<int64_t> strides);

// Shift a var at pos by a constant offset in an IntegerRelation, i.e. replace
// var with var' = var + offset.
presburger::IntegerRelation shiftVar(
    const presburger::IntegerRelation& relation, unsigned int pos,
    int64_t offset);

// Get layout relation that corresponds to a tensor::extract_slice op.
FailureOr<presburger::IntegerRelation> getSliceExtractionRelation(
    RankedTensorType sourceType, RankedTensorType resultType,
    SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
    SmallVector<int64_t> strides);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_UTILS_H_
