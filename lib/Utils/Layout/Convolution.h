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

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication.
FailureOr<presburger::IntegerRelation> get2dConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize);

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

RankedTensorType get2dConvChwFchwFilterExpandedType(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    ArrayRef<int64_t> strides = {1, 1});

// Returns an IntegerRelation that represents a diagonalized 2-D Toeplitz matrix
// that is used to compute a 2-D multichannel convolution filter such that the
// convolution is equivalent a matrix product with the flattened multichannel
// input vector. Each row corresponds to one filter multiplication. The filter
// type is assumed to be 4-D with dimensions (f, c, h, w) and the data type is
// assumed to be 3-D with dimensions (c, h, w).
FailureOr<presburger::IntegerRelation>
get2dConvChwFchwFilterDiagonalizedRelation(RankedTensorType filterType,
                                           RankedTensorType dataType,
                                           ArrayRef<int64_t> strides,
                                           int64_t padding,
                                           int64_t ciphertextSize,
                                           bool interchangeRows = true);

// Returns an IntegerRelation for a row-interchange map that optimizes the
// diagonal structure of a convolution's Toeplitz matrix.
//
// It maps flattened indices from a channel-last (H, W, C*g^2) tensor to a
// (gH, gW, C) tensor. This rearrangement interleaves sub-pixels
// from the channel dimension into g x g spatial blocks, effectively performing
// a depth-to-space (pixel-shuffle) operation.
// See Orion's implementation of multiplex:
// https://github.com/baahl-nyu/orion/blob/0f7df1717be44e21caeab42f8a9da81c997fe7e8/orion/core/packing.py#L159
// This computes the flattened input to flattened output map, e.g.
// input = torch.arange(n * c * h * w).reshape(n, c, h, w)
// result = multiplex(input, gap)
// flattened_result = result.squeeze(0).flatten()
presburger::IntegerRelation getRowInterchangeRelation(int64_t c, int64_t h,
                                                      int64_t w, int64_t g);

bool isRelation2dConvFilterDiagonalized(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize, const presburger::IntegerRelation& relation);

// Returns an IntegerRelation that corresponds to the output layout of a 2-D
// multi-channel convolution. This includes the row interchange from pixel
// shuffling. The result is a relation mapping to (ct, slot) of the output.
presburger::IntegerRelation get2dConvResultRelation(
    RankedTensorType outputType, ArrayRef<int64_t> strides, int64_t padding,
    int64_t ciphertextSize, bool interchangeRows = true);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_H_
