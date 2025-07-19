
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "pocketfft_hdronly.h"                        // from @pocketfft

namespace mlir {
namespace heir {
namespace approximation {

using ::llvm::APFloat;
using ::llvm::SmallVector;
using polynomial::FloatPolynomial;

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

// Port of `np.maximum.accumulate(b[::-1])[::-1]`
SmallVector<APFloat> accumulateMaximumReverse(SmallVector<APFloat> b) {
  SmallVector<APFloat> result;
  if (b.empty()) {
    return result;
  }
  result.resize(b.size(), APFloat(0.0));
  APFloat currentMax = b.back();
  result.back() = currentMax;
  for (int i = b.size() - 2; i >= 0; --i) {
    if (b[i] > currentMax) {
      currentMax = b[i];
    }
    result[i] = currentMax;
  }
  return result;
}

/// Ported from https://github.com/chebfun/chebfun/blob/master/standardChop.m
/// chops COEFFS at a point beyond which it is smaller than tol^(2/3).
/// coeffs will never be chopped unless it is of length at least 17 and falls at
/// least below TOL^(1/3). It will always be chopped if it has a long enough
/// final segment below TOL, and the final entry COEFFS(CUTOFF) will never
/// be smaller than TOL^(7/6).  All these statements are relative to
/// MAX(ABS(COEFFS)) and assume CUTOFF > 1.  These parameters result from
/// extensive experimentation involving functions such as those presented in
/// the paper cited above.  They are not derived from first principles and
/// there is no claim that they are optimal.
size_t standardChop(SmallVector<APFloat> coeffs, double tol) {
  if (tol >= 1) return 1;

  size_t n = coeffs.size();
  if (n < 17) return n;

  // Step 1: Convert COEFFS to a new monotonically nonincreasing
  //         vector ENVELOPE normalized to begin with the value 1.
  SmallVector<APFloat> absCoeffs =
      llvm::map_to_vector(coeffs, [](APFloat x) { return abs(x); });
  SmallVector<APFloat> envelope = accumulateMaximumReverse(absCoeffs);
  if (envelope[0].isZero()) return 1;

  // Normalize the envelope.
  for (int i = 0; i < envelope.size(); ++i) {
    envelope[i] = envelope[i] / envelope[0];
  }

  // Step 2: Scan ENVELOPE for a value PLATEAUPOINT, the first point J-1, if
  // any, that is followed by a plateau.  A plateau is a stretch of coefficients
  // ENVELOPE(J),...,ENVELOPE(J2), J2 = round(1.25*J+5) <= N, with the property
  // that ENVELOPE(J2)/ENVELOPE(J) > R.  The number R ranges from R = 0 if
  // ENVELOPE(J) = TOL up to R = 1 if ENVELOPE(J) = TOL^(2/3).  Thus a potential
  // plateau whose starting value is ENVELOPE(J) ~ TOL^(2/3) has to be perfectly
  // flat to count, whereas with ENVELOPE(J) ~ TOL it doesn't have to be flat at
  // all.  If a plateau point is found, then we know we are going to chop the
  // vector, but the precise chopping point CUTOFF still remains to be
  // determined in Step 3.

  size_t plateauPoint = 0;
  size_t j2 = 0;
  for (size_t j = 2; j < n + 1; ++j) {
    j2 = std::round(1.25 * j + 5);
    if (j2 > n) {
      // there is no plateau: exit
      return coeffs.size();
    }
    APFloat e1 = envelope[j - 1];
    APFloat e2 = envelope[j2 - 2];
    double r = 3 * (1 - std::log(e1.convertToDouble()) / std::log(tol));
    if (e1.isZero() || (e2 / e1).convertToDouble() > r) {
      // a plateau has been found: go to Step 3
      plateauPoint = j - 2;
      break;
    }
  }

  // Step 3: fix CUTOFF at a point where ENVELOPE, plus a linear function
  // included to bias the result towards the left end, is minimal.
  //
  // Some explanation is needed here.  One might imagine that if a plateau is
  // found, then one should simply set CUTOFF = PLATEAUPOINT and be done,
  // without the need for a Step 3. However, sometimes CUTOFF should be smaller
  // or larger than PLATEAUPOINT, and that is what Step 3 achieves.
  //
  // CUTOFF should be smaller than PLATEAUPOINT if the last few coefficients
  // made negligible improvement but just managed to bring the vector ENVELOPE
  // below the level TOL^(2/3), above which no plateau will ever be detected.
  // This part of the code is important for avoiding situations where a
  // coefficient vector is chopped at a point that looks "obviously wrong" with
  // PLOTCOEFFS.
  //
  // CUTOFF should be larger than PLATEAUPOINT if, although a plateau has been
  // found, one can nevertheless reduce the amplitude of the coefficients a good
  // deal further by taking more of them.  This will happen most often when a
  // plateau is detected at an amplitude close to TOL, because in this case, the
  // "plateau" need not be very flat.  This part of the code is important to
  // getting an extra digit or two beyond the minimal prescribed accuracy when
  // it is easy to do so.

  if (envelope[plateauPoint].isZero()) return plateauPoint;

  size_t j3 = 0;
  // port of np.sum(envelope >= tol ** (7 / 6))
  double tolPow = std::pow(tol, 7.0 / 6.0);
  for (size_t i = 0; i < envelope.size(); ++i) {
    if (envelope[i].convertToDouble() >= tolPow) {
      ++j3;
    }
  }
  if (j3 < j2) {
    j2 = j3 + 1;
    envelope[j2] = APFloat(std::pow(tol, 7.0 / 6.0));
  }

  SmallVector<double> cc;
  cc.reserve(j2);
  for (size_t i = 0; i < j2; ++i) {
    cc.push_back(std::log10(envelope[i].convertToDouble()));
  }

  // port of cc = cc + np.linspace(0, (-1 / 3) * np.log10(tol), j2)
  double linspace = (-1.0 / 3.0) * std::log10(tol);
  for (size_t i = 0; i < j2; ++i) {
    cc[i] += linspace * i / (j2 - 1);
  }

  // port of d = np.argmin(cc)
  size_t d = 0;
  double min_cc = cc[0];
  for (size_t i = 1; i < cc.size(); ++i) {
    if (cc[i] < min_cc) {
      min_cc = cc[i];
      d = i;
    }
  }

  size_t cutoff = std::max(d, (size_t)1);
  return cutoff;
}

void interpolateChebyshevWithSmartDegreeSelection(
    const std::function<APFloat(APFloat)> &func,
    SmallVector<APFloat> &outputChebCoeffs, double tolerance,
    int64_t maxDegree) {
  int64_t deg = 17;
  SmallVector<APFloat> chebPts, chebEvalPts, chebCoeffs;
  while (deg <= maxDegree) {
    int32_t numChebPts = 1 + deg;
    chebPts.clear();
    chebEvalPts.clear();
    chebCoeffs.clear();
    chebPts.reserve(numChebPts);
    getChebyshevPoints(numChebPts, chebPts);
    for (auto &pt : chebPts) {
      chebEvalPts.push_back(func(pt));
    }
    interpolateChebyshev(chebEvalPts, chebCoeffs);

    size_t cutoff = standardChop(chebCoeffs, tolerance);
    if (cutoff < deg) {
      outputChebCoeffs.reserve(cutoff);
      // Copy the coefficients up to the cutoff point.
      for (int i = 0; i < cutoff; ++i) {
        outputChebCoeffs.push_back(chebCoeffs[i]);
      }
      return;
    }
    deg = 2 * deg - 1;
  }
  outputChebCoeffs.reserve(chebCoeffs.size());
  for (auto &coeff : chebCoeffs) {
    outputChebCoeffs.push_back(coeff);
  }
}

}  // namespace approximation
}  // namespace heir
}  // namespace mlir
