#ifndef LIB_UTILS_APPROXIMATION_CHEBYSHEV_H_
#define LIB_UTILS_APPROXIMATION_CHEBYSHEV_H_

#include <cstdint>

#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {

/// Generate Chebyshev points of the second kind, storing them in the results
/// outparameter. The output points are ordered left to right on the interval
/// [-1, 1].
///
/// This is a port of the chebfun routine at
/// https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/%40chebtech2/chebpts.m#L34
void getChebyshevPoints(int64_t numPoints,
                        SmallVector<::llvm::APFloat> &results);

/// Generate the first `numPolynomials` Chebyshev polynomials of the second
/// kind, storing them in the results outparameter.
///
/// The first few polynomials are 1, 2x, 4x^2 - 1, 8x^3 - 4x, ...
void getChebyshevPolynomials(
    int64_t numPolynomials,
    SmallVector<::mlir::heir::polynomial::FloatPolynomial> &results);

}  // namespace approximation
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APPROXIMATION_CHEBYSHEV_H_
