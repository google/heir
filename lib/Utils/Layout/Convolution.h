#ifndef LIB_UTILS_LAYOUT_CONVOLUTION_H_
#define LIB_UTILS_LAYOUT_CONVOLUTION_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/include/llvm/ADT/DenseSet.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

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

RankedTensorType get1dConvCwFcwFilterExpandedType(RankedTensorType filterType,
                                                  RankedTensorType dataType,
                                                  int64_t stride,
                                                  int64_t padding);

RankedTensorType get2dConvChwFchwFilterExpandedType(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    ArrayRef<int64_t> strides = {1, 1});

// Returns an IntegerRelation that represents a diagonalized 2-D Toeplitz matrix
// that is used to compute a 1-D multichannel convolution filter such that the
// convolution is equivalent a matrix product with the flattened multichannel
// input vector. Each row corresponds to one filter multiplication. The filter
// type is assumed to be 3-D with dimensions (f, c, w) and the data type is
// assumed to be 3-D with dimensions (1, c, w).
// `dataSlotPermutation`, when non-null, maps the flattened data index [j] to
// the [ct, slot] the element actually occupies. It re-indexes the matrix's
// column space by that packing, so the diagonal kernel reads the data where it
// already sits instead of converting the ciphertext. It must give each column
// at most one slot and no two columns the same slot; see
// getDiagonalColumnRepresentative. A column with no slot is dropped, which is
// correct exactly when that element is zero.
FailureOr<presburger::IntegerRelation> get1dConvCwFcwFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t stride,
    int64_t padding, int64_t minSlotCount, bool interchangeRows = true,
    const presburger::IntegerRelation* dataSlotPermutation = nullptr);

// Flattens a 3-D (1, C, W) data layout `[n, c, w] -> [ct, slot]` into the
// column-space permutation `[j] -> [ct, slot]` with j = c * W + w, as accepted
// by `get1dConvCwFcwFilterDiagonalizedRelation`'s `dataSlotPermutation`.
//
// `matrixDataType` is the operand the Toeplitz matrix is built against, so it
// fixes W and therefore the column space. `padding` is the padding the matrix
// carries in its own `padding` parameter. It is nonzero when a `tensor.pad`
// folded into the conv: the matrix is then built against the unpadded operand
// while `dataLayout` still indexes the padded value, so column j must read the
// slot of padded index (c, w + padding).
//
// Fails if the layout does not pack the data into ciphertext zero, and, when
// `padding` is nonzero, if the shifted window leaves any column without a slot.
// Every column is real data in that case, so dropping one would drop data.
FailureOr<presburger::IntegerRelation> get1dConvDataColumnPermutation(
    RankedTensorType matrixDataType,
    const presburger::IntegerRelation& dataLayout, int64_t padding = 0);

// The columns j = c * W + w that `columnPermutation` gives a slot to read. The
// diagonal kernel drops any column outside this set from the plaintext matrix,
// so a caller that absorbs a packing must check that those elements are zero.
llvm::DenseSet<int64_t> getMappedConvMatrixColumns(
    const presburger::IntegerRelation& columnPermutation);

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
presburger::IntegerRelation get2dConvResultRelation(RankedTensorType outputType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding,
                                                    int64_t minSlotCount);

presburger::IntegerRelation get2dConvRowInterchangeLayoutRelation(
    RankedTensorType outputType, ArrayRef<int64_t> strides,
    int64_t minSlotCount);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_H_
