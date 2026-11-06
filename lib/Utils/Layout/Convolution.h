#ifndef LIB_UTILS_LAYOUT_CONVOLUTION_H_
#define LIB_UTILS_LAYOUT_CONVOLUTION_H_

#include <cstdint>
#include <vector>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// A gapped (pixel-shuffled) convolution layout folds a block of output channels
// into each spatial block: `gap * gap` channels for a 2-D conv, `gap` channels
// for a 1-D conv. That shuffle is a bijection only when the channel count is a
// multiple of the block, so a layout reserves the next multiple and leaves the
// extra channels empty. The expanded filter matrix has zero rows for them, and
// the result layout maps nothing into their slots.
int64_t getPaddedConvChannels(int64_t outputChannels, int64_t channelsPerBlock);

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication. This does not include diagonalizing
// the matrix, the returned relation only expands the filter to the data matrix.
presburger::IntegerRelation get2dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding);

// Returns an IntegerRelation that expands a 1-D filter used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the input vector. Each row
// corresponds to one filter multiplication. This does not include diagonalizing
// the matrix, the returned relation only expands the filter to the data matrix.
presburger::IntegerRelation get1dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    int64_t stride,
                                                    int64_t padding);

RankedTensorType get2dConvFilterExpandedType(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    ArrayRef<int64_t> strides = {1, 1});

RankedTensorType get1dConvFilterExpandedType(RankedTensorType filterType,
                                             RankedTensorType dataType,
                                             int64_t stride, int64_t padding);

// Returns an IntegerRelation that expands a filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication.
FailureOr<presburger::IntegerRelation> getConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t minSlotCount);

// Returns an IntegerRelation that expands a multichannel filter used
// in a 2-D convolution into a 2-D Toeplitz matrix such that the convolution is
// equivalent a matrix product with the flattened multichannel input vector.
// Each row corresponds to one filter multiplication. This does not include
// diagonalizing the matrix, this simply returns the expanded data matrix. The
// filter type is assumed to be 4-D with dimensions (f, c, h, w) and the data
// type is assumed to be 3-D within a 4-D tensor of dimensions (1, c, h, w).
presburger::IntegerRelation get2dConvChwFchwFilterRelation(
    RankedTensorType filterType, RankedTensorType dataType,
    ArrayRef<int64_t> strides, int64_t padding);

// Returns an IntegerRelation that expands a multichannel filter used
// in a 1-D convolution into a 2-D Toeplitz matrix such that the convolution is
// equivalent a matrix product with the flattened multichannel input vector.
// Each row corresponds to one filter multiplication. This does not include
// diagonalizing the matrix, this simply returns the expanded data matrix. The
// filter type is assumed to be 3-D with dimensions (f, c, w) and the data
// type is assumed to be 2-D with dimensions (1, c, w).
presburger::IntegerRelation get1dConvCwFcwFilterRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t stride,
    int64_t padding);

// `interchangeRows` must match the flag the filter layout was built with: an
// interchanged (pixel-shuffled) layout reserves whole channel blocks, so its
// matrix has extra zero rows when the channel count is not a multiple of the
// block. The Halevi-Shoup kernel is sized from this type, so it has to agree
// with the layout relation.
RankedTensorType get1dConvCwFcwFilterExpandedType(RankedTensorType filterType,
                                                  RankedTensorType dataType,
                                                  int64_t stride,
                                                  int64_t padding,
                                                  bool interchangeRows = true);

RankedTensorType get2dConvChwFchwFilterExpandedType(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    ArrayRef<int64_t> strides = {1, 1}, bool interchangeRows = true);

// Returns an IntegerRelation that represents a diagonalized 2-D Toeplitz matrix
// that is used to compute a 1-D multichannel convolution filter such that the
// convolution is equivalent a matrix product with the flattened multichannel
// input vector. Each row corresponds to one filter multiplication. The filter
// type is assumed to be 3-D with dimensions (f, c, w) and the data type is
// assumed to be 3-D with dimensions (1, c, w).
FailureOr<presburger::IntegerRelation> get1dConvCwFcwFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t stride,
    int64_t padding, int64_t minSlotCount, bool interchangeRows = true);

// Returns a sequence of IntegerRelations that represents the layout mapping as
// a series of simple steps (Toeplitz expansion, row interchange, flattening,
// diagonalization). This is preferred for compilation performance to avoid ISL
// hangs when generating loops.
FailureOr<std::vector<presburger::IntegerRelation>>
get2dConvChwFchwFilterAsSequence(RankedTensorType filterType,
                                 RankedTensorType dataType,
                                 ArrayRef<int64_t> strides, int64_t padding,
                                 int64_t minSlotCount,
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
presburger::IntegerRelation get2dConvRowInterchangeRelation(int64_t c,
                                                            int64_t h,
                                                            int64_t w,
                                                            int64_t g);

// Returns an IntegerRelation for a row-interchange map that optimizes the
// diagonal structure of a convolution's Toeplitz matrix.
presburger::IntegerRelation get1dConvRowInterchangeRelation(int64_t c,
                                                            int64_t w,
                                                            int64_t g);

bool isRelationConvFilterDiagonalized(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t minSlotCount, const presburger::IntegerRelation& relation);

// Returns an IntegerRelation that corresponds to the output layout of a 1-D
// multi-channel convolution. This includes the row interchange from pixel
// shuffling. The result is a relation mapping to (ct, slot) of the output.
presburger::IntegerRelation get1dConvResultRelation(
    RankedTensorType outputType, int64_t stride, int64_t padding,
    int64_t minSlotCount, bool interchangeRows = true);

// Returns an IntegerRelation that corresponds to the output layout of a 2-D
// multi-channel convolution. This includes the row interchange from pixel
// shuffling. The result is a relation mapping to (ct, slot) of the output.
//
// Set `interchangeRows` when the caller composes this with
// get2dConvRowInterchangeLayoutRelation: a shuffled result reserves whole
// channel blocks, and both relations must use that same larger extent as the
// replication period.
presburger::IntegerRelation get2dConvResultRelation(
    RankedTensorType outputType, ArrayRef<int64_t> strides, int64_t padding,
    int64_t minSlotCount, bool interchangeRows = false);

presburger::IntegerRelation get2dConvRowInterchangeLayoutRelation(
    RankedTensorType outputType, ArrayRef<int64_t> strides,
    int64_t minSlotCount);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_H_
