#include "lib/Utils/Approximation/CaratheodoryFejer.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>

#include "Eigen/Core"         // from @eigen
#include "Eigen/Dense"        // from @eigen
#include "Eigen/Eigenvalues"  // from @eigen
#include "lib/Utils/Approximation/Chebyshev.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {

using ::Eigen::MatrixXd;
using ::Eigen::SelfAdjointEigenSolver;
using ::Eigen::VectorXd;
using ::llvm::APFloat;
using ::llvm::SmallVector;
using polynomial::FloatPolynomial;

FloatPolynomial caratheodoryFejerApproximationUnitInterval(
    const std::function<APFloat(APFloat)>& func, int32_t degree) {
  // Construct the Chebyshev interpolant.
  SmallVector<APFloat> chebCoeffs;
  interpolateChebyshevWithSmartDegreeSelection(func, chebCoeffs);
  size_t chebDegree = chebCoeffs.size() - 1;
  if (chebDegree <= degree) return chebyshevToMonomial(chebCoeffs);

  // Use the tail coefficients to construct a Hankel matrix
  // where A[i, j] = c[i+j]
  // Cf. https://en.wikipedia.org/wiki/Hankel_matrix
  SmallVector<APFloat> tailChebCoeffs(chebCoeffs.begin() + (degree + 1),
                                      chebCoeffs.end());
  int32_t hankelSize = tailChebCoeffs.size();
  MatrixXd hankel(hankelSize, hankelSize);
  for (int i = 0; i < hankelSize; ++i) {
    for (int j = 0; j < hankelSize; ++j) {
      // upper left triangular region, including diagonal
      if (i + j < hankelSize)
        hankel(i, j) = tailChebCoeffs[i + j].convertToDouble();
      else
        hankel(i, j) = 0;
    }
  }

  // Compute the eigenvalues and eigenvectors of the Hankel matrix
  SelfAdjointEigenSolver<MatrixXd> solver(hankel);

  const VectorXd& eigenvalues = solver.eigenvalues();
  // Eigenvectors are columns of the matrix.
  const MatrixXd& eigenvectors = solver.eigenvectors();

  // Extract the eigenvector for the (absolute value) largest eigenvalue.
  int32_t maxIndex = 0;
  double maxEigenvalue = std::abs(eigenvalues(0));
  for (int32_t i = 1; i < eigenvalues.size(); ++i) {
    if (std::abs(eigenvalues(i)) > maxEigenvalue) {
      maxEigenvalue = std::abs(eigenvalues(i));
      maxIndex = i;
    }
  }
  VectorXd maxEigenvector = eigenvectors.col(maxIndex);

  // A debug for comparing the eigenvalue solver with the reference
  // implementation.
  // std::cout << "Max eigenvector:" << std::endl;
  // for (int32_t i = 0; i < maxEigenvector.size(); ++i) {
  //   std::cout << std::setprecision(18) << maxEigenvector(i) << ", ";
  // }
  // std::cout << std::endl;

  double v1 = maxEigenvector(0);
  VectorXd vv = maxEigenvector.tail(maxEigenvector.size() - 1);

  SmallVector<APFloat> b =
      SmallVector<APFloat>(tailChebCoeffs.begin(), tailChebCoeffs.end());

  int32_t t = chebDegree - degree - 1;
  for (int32_t i = degree; i > -degree - 1; --i) {
    SmallVector<APFloat> sliceB(b.begin(), b.begin() + t);

    APFloat sum = APFloat(0.0);
    for (int32_t j = 0; j < sliceB.size(); ++j) {
      double vvVal = vv(j);
      sum = sum + sliceB[j] * APFloat(vvVal);
    }

    APFloat z = -sum / APFloat(v1);

    // I suspect this insert is slow. Once it's working we can optimize this
    // loop to avoid the insert.
    b.insert(b.begin(), z);
  }

  SmallVector<APFloat> bb(b.begin() + degree, b.begin() + (2 * degree + 1));
  for (int32_t i = 1; i < bb.size(); ++i) {
    bb[i] = bb[i] + b[degree - 1 - (i - 1)];
  }

  SmallVector<APFloat> pk;
  pk.reserve(bb.size());
  for (int32_t i = 0; i < bb.size(); ++i) {
    pk.push_back(chebCoeffs[i] - bb[i]);
  }

  return chebyshevToMonomial(pk);
}

FloatPolynomial caratheodoryFejerApproximation(
    const std::function<APFloat(APFloat)>& func, int32_t degree, double lower,
    double upper) {
  bool needsScaling = lower != -1.0 || upper != 1.0;
  double midPoint = (lower + upper) / 2;
  double halfLen = (upper - lower) / 2;
  std::function<APFloat(APFloat)> funcScaled;
  if (needsScaling) {
    funcScaled = [&](const APFloat& x) {
      APFloat input = APFloat(midPoint) + APFloat(halfLen) * x;
      return func(input);
    };
  } else {
    funcScaled = func;
  }
  FloatPolynomial approximant =
      caratheodoryFejerApproximationUnitInterval(funcScaled, degree);

  if (needsScaling) {
    FloatPolynomial y =
        FloatPolynomial::fromCoefficients({-midPoint / halfLen, 1 / halfLen});
    approximant = approximant.compose(y);
  }
  return approximant;
}

}  // namespace approximation
}  // namespace heir
}  // namespace mlir
