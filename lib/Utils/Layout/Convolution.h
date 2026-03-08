#ifndef LIB_UTILS_LAYOUT_CONVOLUTION_H_
#define LIB_UTILS_LAYOUT_CONVOLUTION_H_

#include <cstdint>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication. This does not include diagonalizing
// the matrix, this simply returns the expanded data matrix.
presburger::IntegerRelation get2dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding);

RankedTensorType get2dConvFilterExpandedType(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    ArrayRef<int64_t> strides = {1, 1});

// Returns an IntegerRelation that expands a multichannel filter used
// in a 2-D convolution into a 2-D Toeplitz matrix such that the convolution is
// equivalent a matrix product with the flattened multichannel input vector.
// Each row corresponds to one filter multiplication. This does not include
// diagonalizing the matrix, this simply returns the expanded data matrix. The
// filter type is assumed to be 4-D with dimensions (f, c, h, w) and the data
// type is assumed to be 3-D with dimensions (c, h, w).
presburger::IntegerRelation get2dConvChwFchwFilterRelation(
    RankedTensorType filterType, RankedTensorType dataType,
    ArrayRef<int64_t> strides, int64_t padding);

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication.
FailureOr<presburger::IntegerRelation> get2dConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize);

bool isRelation2dConvFilterDiagonalized(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize, const presburger::IntegerRelation& relation);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_H_
