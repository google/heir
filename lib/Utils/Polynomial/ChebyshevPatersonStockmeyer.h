#ifndef LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_
#define LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Utils/Polynomial/ChebyshevDecomposition.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

// The minimum absolute value of a coefficient to be considered in the
// evaluation of the Chebyshev polynomial.
constexpr double kMinCoeffs = 1e-15;

// Computes Arithmetic DAGs of x^0, x^1, ..., x^k.
// The multiplicative depth is ceil(log2(k)).
template <typename T>
std::vector<std::shared_ptr<kernel::ArithmeticDagNode<T>>> computePowers(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x, int64_t k) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  std::vector<std::shared_ptr<NodeTy>> result;
  result.reserve(k + 1);
  result.push_back(NodeTy::constantScalar(1));
  if (k >= 1) {
    result.push_back(x);
  }
  for (int64_t i = 2; i <= k; i++) {
    if (i % 2 == 0) {
      // X^{2n} = X^n * X^n
      result.push_back(NodeTy::mul(result[i / 2], result[i / 2]));
    } else {
      // X^{2n+1} = X^n * X^{n+1}
      result.push_back(NodeTy::mul(result[i / 2], result[i / 2 + 1]));
    }
  }
  return result;
}

// Computes Arithmetic DAGs of T_0(x), T_1(x), ..., T_k(x), where T_i are
// Chebyshev polynomials.
// The multiplicative depth is ceil(log2(k)).
template <typename T>
std::enable_if_t<std::is_base_of<kernel::AbstractValue, T>::value,
                 std::vector<std::shared_ptr<kernel::ArithmeticDagNode<T>>>>
computeChebyshevPolynomialValues(const T& x, int64_t k) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  std::vector<std::shared_ptr<NodeTy>> result;
  result.reserve(k + 1);
  auto number1 = NodeTy::constantScalar(1);
  auto number2 = NodeTy::constantScalar(2);
  auto xNode = NodeTy::leaf(x);
  result.push_back(number1);
  if (k >= 1) {
    result.push_back(xNode);
  }
  for (int64_t i = 2; i <= k; i++) {
    if (i % 2 == 0) {
      // T_{2n}(x) = 2(T_n(x))^2 - 1
      result.push_back(NodeTy::sub(
          NodeTy::mul(NodeTy::mul(result[i / 2], result[i / 2]), number2),
          number1));
    } else {
      // T_{2n+1}(x) = 2*T_n(x) * T_{n+1}(x) - x
      result.push_back(NodeTy::sub(
          NodeTy::mul(NodeTy::mul(result[i / 2], result[i / 2 + 1]), number2),
          xNode));
    }
  }
  return result;
}

// Returns true if any element in `v` has an absolute value greater than or
// equal to `threshold`.
inline bool hasElementsLargerThan(const std::vector<double>& v,
                                  double threshold) {
  return std::any_of(v.begin(), v.end(), [threshold](double a) {
    return std::abs(a) >= threshold;
  });
}

// Creates Arithmetic DAG of evaluation a Chebyshev polynomial with
// PatersonStockmeyer's algorithm.
template <typename T>
std::enable_if_t<std::is_base_of<kernel::AbstractValue, T>::value,
                 std::shared_ptr<kernel::ArithmeticDagNode<T>>>
patersonStockmeyerChebyshevPolynomialEvaluation(
    const T& x, ::llvm::ArrayRef<double> coefficients,
    double minCoeffThreshold = kMinCoeffs) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  int64_t polynomialDegree = coefficients.size() - 1;
  // Choose k optimally - sqrt of maxDegree is typically a good choice
  int64_t k =
      std::max(static_cast<int64_t>(std::ceil(std::sqrt(polynomialDegree))),
               static_cast<int64_t>(1));

  // Decompose p = coeffs[0] + coeffs[1]*T_k + coeffs[2]*T_k^2 + ... +
  // coeffs[l]*T_k^l.
  polynomial::ChebyshevDecomposition decomposition =
      polynomial::decompose(coefficients, k);

  // Precompute T_0(x), T_1(x), ..., T_k(x).
  std::vector<std::shared_ptr<NodeTy>> chebPolynomialValues =
      computeChebyshevPolynomialValues(x, k);

  // Precompute (T_k(x))^0, (T_k(x))^1, ..., (T_k(x))^l.
  int64_t l = decomposition.coeffs.size() - 1;
  std::vector<std::shared_ptr<NodeTy>> chebKPolynomialPowers =
      computePowers(chebPolynomialValues.back(), l);

  // Evaluate the polynomial.
  std::shared_ptr<NodeTy> result;
  for (int i = 0; i < decomposition.coeffs.size(); ++i) {
    if (!hasElementsLargerThan(decomposition.coeffs[i], minCoeffThreshold))
      continue;
    std::shared_ptr<NodeTy> pol;
    for (int j = 0; j < decomposition.coeffs[i].size(); ++j) {
      double coeff = decomposition.coeffs[i][j];
      // Skip coefficients that are too small.
      if (std::abs(coeff) < minCoeffThreshold) continue;

      auto coefNode = NodeTy::constantScalar(coeff);
      auto termNode = NodeTy::mul(coefNode, chebPolynomialValues[j]);
      if (pol) {
        pol = NodeTy::add(pol, termNode);
      } else {
        pol = termNode;
      }
    }
    if (!pol) continue;
    if (i > 0) {
      pol = NodeTy::mul(pol, chebKPolynomialPowers[i]);
    }
    if (result) {
      result = NodeTy::add(result, pol);
    } else {
      result = pol;
    }
  }
  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_
