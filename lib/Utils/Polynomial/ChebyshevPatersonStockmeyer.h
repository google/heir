#ifndef LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_
#define LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/ChebyshevDecomposition.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

// The minimum absolute value of a coefficient to be considered in the
// evaluation of the Chebyshev polynomial.
constexpr double kMinCoeffs = 1e-15;

// A C++ port of Lattigo's `bignum.OptimalSplit` function.
// This function determines the optimal split point for the Paterson-Stockmeyer
// algorithm by calculating the number of multiplications for two possible
// split points and choosing the better one.
//
// Args:
//   logDegree: ceil(log2(degree)) of the polynomial.
//
// Returns:
//   The optimal split parameter (log scale).
inline int64_t optimalSplit(int64_t logDegree) {
  int64_t logSplit = logDegree >> 1;
  int64_t a = (1LL << logSplit) + (1LL << (logDegree - logSplit)) + logDegree -
              logSplit - 3;
  int64_t b = (1LL << (logSplit + 1)) + (1LL << (logDegree - logSplit - 1)) +
              logDegree - logSplit - 4;
  if (a > b) {
    logSplit += 1;
  }
  return logSplit;
}

// Returns a + b = n such that |a-b| is minimized.
//
// For Chebyshev basis, tries to keep a and/or b odd if possible to
// maximize the number of odd terms.
//
// Based on [Lee et al. 2020]: High-Precision and Low-Complexity Approximate
// Homomorphic Encryption by Error Variance Minimization
//
// Args:
//   n: The degree to split
//
// Returns:
//   Pair (a, b) where a + b = n and |a-b| is minimized
inline std::pair<int64_t, int64_t> splitDegree(int64_t n) {
  if (n <= 0) {
    throw std::invalid_argument("n should be greater than zero");
  }

  // Check if n is a power of 2
  if ((n & (n - 1)) == 0) {
    // Necessary for optimal depth
    return {n / 2, n / 2};
  } else {
    // [Lee et al. 2020] : Maximize the number of odd terms of Chebyshev basis
    // Find k = floor(log2(n-1))
    int64_t k = 0;
    int64_t temp = n - 1;
    while (temp > 1) {
      temp >>= 1;
      k++;
    }
    int64_t a = (1LL << k) - 1;
    int64_t b = n + 1 - (1LL << k);
    return {a, b};
  }
}

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

// Recursively computes T_n(x) for the Chebyshev polynomial T_n.
//
// Uses the recurrence relation for Chebyshev polynomials:
// T_n(x) = 2*T_a(x)*T_b(x) - T_c(x)
// where n = a+b and c = |a-b|
//
// Args:
//   x: The input node representing the variable
//   n: The degree of the Chebyshev polynomial to compute
//   cache: Map caching already computed powers
//
// Returns:
//   Node representing T_n(x)
template <typename T>
std::shared_ptr<kernel::ArithmeticDagNode<T>> genChebyshevPowerRecursive(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x, int64_t n,
    std::map<int64_t, std::shared_ptr<kernel::ArithmeticDagNode<T>>>& cache) {
  using NodeTy = kernel::ArithmeticDagNode<T>;

  if (n == 0) {
    // T_0(x) = 1
    return NodeTy::constantScalar(1);
  }

  if (n == 1) {
    // T_1(x) = x
    // Cache it for consistency
    if (cache.find(1) == cache.end()) {
      cache[1] = x;
    }
    return x;
  }

  // Check cache
  auto it = cache.find(n);
  if (it != cache.end()) {
    return it->second;
  }

  // Split the degree optimally
  auto [a, b] = splitDegree(n);

  // Recursively compute T_a(x) and T_b(x)
  auto t_a = genChebyshevPowerRecursive(x, a, cache);
  auto t_b = genChebyshevPowerRecursive(x, b, cache);

  // Compute T_n(x) = T_a(x) * T_b(x)
  auto t_n = NodeTy::mul(t_a, t_b);

  // Apply Chebyshev recurrence: T_n = 2*T_a*T_b - T_c
  // where c = |a - b|
  int64_t c = std::abs(a - b);

  // Compute T_n = 2*T_a*T_b
  auto two = NodeTy::constantScalar(2);
  t_n = NodeTy::mul(two, t_n);

  // Compute T_n = 2*T_a*T_b - T_c
  if (c == 0) {
    // T_0 = 1, so subtract 1
    t_n = NodeTy::sub(t_n, NodeTy::constantScalar(1));
  } else {
    // Recursively compute T_c
    auto t_c = genChebyshevPowerRecursive(x, c, cache);
    t_n = NodeTy::sub(t_n, t_c);
  }

  // Cache the result
  cache[n] = t_n;

  return t_n;
}

// Generates Chebyshev polynomials T_0(x), T_1(x), ..., T_maxDegree(x).
//
// Args:
//   x: The input node
//   maxDegree: Maximum degree to compute
//
// Returns:
//   Map from degree to Node representing T_degree(x)
template <typename T>
std::map<int64_t, std::shared_ptr<kernel::ArithmeticDagNode<T>>>
genChebyshevPowersRecursive(std::shared_ptr<kernel::ArithmeticDagNode<T>> x,
                            int64_t maxDegree) {
  std::map<int64_t, std::shared_ptr<kernel::ArithmeticDagNode<T>>> cache;

  // Generate all powers up to maxDegree
  for (int64_t i = 1; i <= maxDegree; i++) {
    genChebyshevPowerRecursive(x, i, cache);
  }

  cache[0] = kernel::ArithmeticDagNode<T>::constantScalar(1);
  return cache;
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
  if (polynomialDegree < 0) {
    return nullptr;
  }

  if (polynomialDegree == 0) {
    if (std::abs(coefficients[0]) < minCoeffThreshold) {
      return nullptr;
    }
    return NodeTy::constantScalar(coefficients[0]);
  }

  // Choose k optimally using Lattigo's optimal split
  int64_t logDegree = std::bit_width(static_cast<uint64_t>(polynomialDegree));
  int64_t logSplit = optimalSplit(logDegree);
  int64_t k = 1LL << logSplit;

  // Decompose p = coeffs[0] + coeffs[1]*T_k + coeffs[2]*T_k^2 + ... +
  // coeffs[l]*T_k^l.
  polynomial::ChebyshevDecomposition decomposition =
      polynomial::decompose(coefficients, k);

  // Precompute T_0(x), T_1(x), ..., T_k(x) using recursive approach.
  auto xNode = NodeTy::leaf(x);
  auto chebPolynomialValuesMap = genChebyshevPowersRecursive(xNode, k);

  // Evaluate the polynomial using a tree-like structure
  std::vector<std::shared_ptr<NodeTy>> babySteps;
  for (const auto& coeffs : decomposition.coeffs) {
    if (!hasElementsLargerThan(coeffs, minCoeffThreshold)) {
      babySteps.push_back(nullptr);
      continue;
    }

    std::shared_ptr<NodeTy> pol;
    for (size_t j = 0; j < coeffs.size(); ++j) {
      if (std::abs(coeffs[j]) < minCoeffThreshold) {
        continue;
      }

      auto termNode = NodeTy::mul(NodeTy::constantScalar(coeffs[j]),
                                  chebPolynomialValuesMap[j]);

      if (pol) {
        pol = NodeTy::add(pol, termNode);
      } else {
        pol = termNode;
      }
    }
    babySteps.push_back(pol);
  }

  auto y = chebPolynomialValuesMap[k];
  auto yPower = y;

  // Combine baby steps in tree-like manner
  while (babySteps.size() > 1) {
    std::vector<std::shared_ptr<NodeTy>> nextBabySteps;
    for (size_t i = 0; i < babySteps.size(); i += 2) {
      if (i + 1 < babySteps.size()) {
        auto pEven = babySteps[i];
        auto pOdd = babySteps[i + 1];

        std::shared_ptr<NodeTy> combined;
        if (pOdd) {
          combined = NodeTy::mul(pOdd, yPower);
          if (pEven) {
            combined = NodeTy::add(pEven, combined);
          }
        } else {
          combined = pEven;
        }
        nextBabySteps.push_back(combined);
      } else {
        nextBabySteps.push_back(babySteps[i]);
      }
    }
    babySteps = nextBabySteps;
    yPower = NodeTy::mul(yPower, yPower);
  }

  return babySteps[0];
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_
