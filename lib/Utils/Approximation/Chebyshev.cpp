
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "pocketfft_hdronly.h"               // from @pocketfft

namespace mlir {
namespace heir {
namespace approximation {

using ::llvm::APFloat;
using ::llvm::SmallVector;
using ::mlir::heir::polynomial::FloatPolynomial;

// When we move to C++20, we can use std::numbers::pi
inline constexpr double kPi = 3.14159265358979323846;

void getChebyshevPoints(int64_t numPoints, SmallVector<APFloat> &results) {
  if (numPoints == 0) {
    return;
  }
  if (numPoints == 1) {
    results.push_back(APFloat(0.));
    return;
  }

  // The values are most simply described as
  //
  //     cos(pi * j / (n-1)) for 0 <= j <= n-1.
  //
  // But to enforce symmetry around the origin---broken by slight numerical
  // inaccuracies---and the left-to-right ordering, we apply the identity
  //
  //     cos(x + pi) = -cos(x) = sin(x - pi/2)
  //
  // to arrive at
  //
  //     sin(pi*j/(n-1) - pi/2) = sin(pi * (2j - (n-1)) / (2(n-1)))
  //
  // An this is equivalent to the formula below, where the range of j is shifted
  // and rescaled from {0, ..., n-1} to {-n+1, -n+3, ..., n-3, n-1}.
  int64_t m = numPoints - 1;
  for (int64_t j = -m; j < m + 1; j += 2) {
    results.push_back(APFloat(std::sin(kPi * j / (2 * m))));
  }
}

void getChebyshevPolynomials(int64_t numPolynomials,
                             SmallVector<FloatPolynomial> &results) {
  if (numPolynomials < 1) return;

  if (numPolynomials >= 1) {
    // 1
    results.push_back(FloatPolynomial::fromCoefficients({1.}));
  }
  if (numPolynomials >= 2) {
    // x
    results.push_back(FloatPolynomial::fromCoefficients({0., 1.}));
  }

  if (numPolynomials <= 2) return;

  for (int64_t i = 2; i < numPolynomials; ++i) {
    auto &last = results.back();
    auto &secondLast = results[results.size() - 2];
    results.push_back(last.monomialMul(1).scale(APFloat(2.)).sub(secondLast));
  }
}

FloatPolynomial chebyshevToMonomial(const SmallVector<APFloat> &coefficients) {
  SmallVector<FloatPolynomial> chebPolys;
  chebPolys.reserve(coefficients.size());
  getChebyshevPolynomials(coefficients.size(), chebPolys);

  FloatPolynomial result = FloatPolynomial::zero();
  for (int64_t i = 0; i < coefficients.size(); ++i) {
    result = result.add(chebPolys[i].scale(coefficients[i]));
  }

  return result;
}

void interpolateChebyshev(ArrayRef<APFloat> chebEvalPoints,
                          SmallVector<APFloat> &outputChebCoeffs) {
  size_t n = chebEvalPoints.size();
  if (n == 0) {
    return;
  }
  if (n == 1) {
    outputChebCoeffs.push_back(chebEvalPoints[0]);
    return;
  }

  // When the function being evaluated has even or odd symmetry, we can get
  // coefficients. In particular, even symmetry implies all odd-numbered
  // Chebyshev coefficients are zero. Odd symmetry implies even-numbered
  // coefficients are zero.
  bool isEven =
      std::equal(chebEvalPoints.begin(), chebEvalPoints.begin() + n / 2,
                 chebEvalPoints.rbegin());

  bool isOdd = true;
  for (int i = 0; i < n / 2; ++i) {
    if (chebEvalPoints[i] != -chebEvalPoints[(n - 1) - i]) {
      isOdd = false;
      break;
    }
  }

  // Construct input to ifft so as to compute a Discrete Cosine Transform
  // The inputs are [v_{n-1}, v_{n-2}, ..., v_0, v_1, ..., v_{n-2}]
  std::vector<std::complex<double>> ifftInput;
  size_t fftLen = 2 * (n - 1);
  ifftInput.reserve(fftLen);
  for (size_t i = n - 1; i > 0; --i) {
    ifftInput.emplace_back(chebEvalPoints[i].convertToDouble());
  }
  for (size_t i = 0; i < n - 1; ++i) {
    ifftInput.emplace_back(chebEvalPoints[i].convertToDouble());
  }

  // Compute inverse FFT using minimal API call to pocketfft. This should be
  // equivalent to numpy.fft.ifft, as it uses pocketfft underneath. It's worth
  // noting here that we're computing the Discrete Cosine Transform (DCT) in
  // terms of a complex Discrete Fourier Transform (DFT), but pocketfft appears
  // to have a built-in `dct` function. It may be trivial to switch to
  // pocketfft::dct, but this was originally based on a reference
  // implementation that did not have access to a native DCT. Migrating to a
  // DCT should only be necessary (a) once the reference implementation is
  // fully ported and tested, and (b) if we determine that there's a
  // performance benefit to using the native DCT. Since this routine is
  // expected to be used in doing relatively low-degree approximations, it
  // probably won't be a problem.
  std::vector<std::complex<double>> ifftResult(fftLen);
  pocketfft::shape_t shape{fftLen};
  pocketfft::stride_t strided{sizeof(std::complex<double>)};
  pocketfft::shape_t axes{0};

  pocketfft::c2c(shape, strided, strided, axes, pocketfft::BACKWARD,
                 ifftInput.data(), ifftResult.data(), 1. / fftLen);

  outputChebCoeffs.clear();
  outputChebCoeffs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    outputChebCoeffs.push_back(APFloat(ifftResult[i].real()));
  }

  // Due to the endpoint behavior of Chebyshev polynomials and the properties
  // of the DCT, the non-endpoint coefficients of the DCT are the Chebyshev
  // coefficients scaled by 2.
  for (int i = 1; i < n - 1; ++i) {
    outputChebCoeffs[i] = outputChebCoeffs[i] * APFloat(2.0);
  }

  // Even/odd corrections
  if (isEven) {
    for (size_t i = 1; i < n; i += 2) {
      outputChebCoeffs[i] = APFloat(0.0);
    }
  }

  if (isOdd) {
    for (size_t i = 0; i < n; i += 2) {
      outputChebCoeffs[i] = APFloat(0.0);
    }
  }
}

}  // namespace approximation
}  // namespace heir
}  // namespace mlir
