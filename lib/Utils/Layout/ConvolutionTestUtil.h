#ifndef LIB_UTILS_LAYOUT_CONVOLUTION_TEST_UTIL_H_
#define LIB_UTILS_LAYOUT_CONVOLUTION_TEST_UTIL_H_

#include <cstdint>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// (Slow, for testing only) Returns a single IntegerRelation that represents a
// diagonalized 2-D Toeplitz matrix that is used to compute a 2-D multichannel
// convolution filter.
FailureOr<presburger::IntegerRelation>
get2dConvChwFchwFilterDiagonalizedRelation(RankedTensorType filterType,
                                           RankedTensorType dataType,
                                           ArrayRef<int64_t> strides,
                                           int64_t padding,
                                           int64_t ciphertextSize,
                                           bool interchangeRows = true);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_TEST_UTIL_H_
