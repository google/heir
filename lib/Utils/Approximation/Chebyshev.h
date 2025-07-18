#ifndef LIB_UTILS_APPROXIMATION_CHEBYSHEV_H_
#define LIB_UTILS_APPROXIMATION_CHEBYSHEV_H_

#include <cstdint>
#include <functional>

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
                        ::llvm::SmallVector<::llvm::APFloat> &results);

/// Generate the first `numPolynomials` Chebyshev polynomials of the first
/// kind, storing them in the results outparameter.
///
/// The first few polynomials are 1, x, 2x^2 - 1, 4x^3 - 3x, ...
void getChebyshevPolynomials(
    int64_t numPolynomials,
    ::llvm::SmallVector<polynomial::FloatPolynomial> &results);

/// Convert a vector of Chebyshev coefficients to the monomial basis. If the
/// Chebyshev polynomials are T_0, T_1, ..., then entry i of the input vector
/// is the coefficient of T_i.
polynomial::FloatPolynomial chebyshevToMonomial(
    const ::llvm::SmallVector<::llvm::APFloat> &coefficients);

/// Interpolate Chebyshev coefficients for a given set of points. The values in
/// chebEvalPoints are assumed to be evaluations of the target function on the
/// first N+1 Chebyshev points of the second kind, where N is the degree of the
/// interpolating polynomial. The produced coefficients are stored in the
/// outparameter outputChebCoeffs.
///
/// A port of chebfun vals2coeffs, cf.
/// https://github.com/chebfun/chebfun/blob/69c12cf75f93cb2f36fd4cfd5e287662cd2f1091/%40ballfun/vals2coeffs.m
/// based on the a trigonometric interpolation via the FFT.
///
/// Cf. Henrici, "Fast Fourier Methods in Computational Complex Analysis"
/// https://doi.org/10.1137/1021093
/// https://people.math.ethz.ch/~hiptmair/Seminars/CONVQUAD/Articles/HEN79.pdf
void interpolateChebyshev(
    ::llvm::ArrayRef<::llvm::APFloat> chebEvalPoints,
    ::llvm::SmallVector<::llvm::APFloat> &outputChebCoeffs);

/// Computes a Chebyshev interpolant of the given function on the unit interval,
/// automatically choosing the degree of the approximation to ensure that (a)
/// the degree is not more than maxDegree, or (b) the absolute error is less
/// than the given tolerance.
void interpolateChebyshevWithSmartDegreeSelection(
    const std::function<APFloat(APFloat)> &func,
    ::llvm::SmallVector<::llvm::APFloat> &outputChebCoeffs,
    double tolerance = 1e-16, int64_t maxDegree = 129);

}  // namespace approximation
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APPROXIMATION_CHEBYSHEV_H_
