#ifndef LIB_UTILS_LAYOUT_CONVOLUTION_TEST_UTIL_H_
#define LIB_UTILS_LAYOUT_CONVOLUTION_TEST_UTIL_H_

#include <cstdint>
#include <vector>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// A dense (N, C, H, W) integer tensor, as the conv test references use it.
using ConvTensor4D = std::vector<std::vector<std::vector<std::vector<int>>>>;

// Number of windows a filter of `filterExtent` takes across one spatial dim of
// `dataExtent`, zero-padded by `padding` on both ends. Shared so that every
// conv reference here agrees with the kernels on the output extents.
int64_t convOutputExtent(int64_t dataExtent, int64_t filterExtent,
                         int64_t stride, int64_t padding);

// A deterministic non-zero (F, C, filterH, filterW) filter. Non-zero
// everywhere, so that an entry landing in the wrong row or column of an
// expanded matrix is visible rather than coincidentally zero.
ConvTensor4D deterministicConvFilter(int64_t outputChannels,
                                     int64_t inputChannels, int64_t filterH,
                                     int64_t filterW);

// Direct 2-D multichannel convolution of `data` zero-padded by `padding` on the
// H and W dims, as an independent reference for the packed kernels.
ConvTensor4D reference2dConv(const ConvTensor4D& data,
                             const ConvTensor4D& filter, int64_t stride,
                             int64_t padding);

// Reference dense expanded Toeplitz matrix for a 2-D multichannel convolution
// with the given stride and a symmetric zero padding of `padding` on both
// spatial dims. Rows are (f, oh, ow) row-major; columns index the *unpadded*
// data as (c, h, w) row-major, so a window position that reaches into the
// padding simply contributes no column.
std::vector<std::vector<int>> reference2dConvChwFchwMatrix(
    const ConvTensor4D& filter, int64_t dataH, int64_t dataW, int64_t stride,
    int64_t padding);

// (Slow, for testing only) Returns a single IntegerRelation that represents a
// diagonalized 2-D Toeplitz matrix that is used to compute a 2-D multichannel
// convolution filter.
FailureOr<presburger::IntegerRelation>
get2dConvChwFchwFilterDiagonalizedRelation(RankedTensorType filterType,
                                           RankedTensorType dataType,
                                           ArrayRef<int64_t> strides,
                                           int64_t padding,
                                           int64_t minSlotCount,
                                           bool interchangeRows = true);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_TEST_UTIL_H_
