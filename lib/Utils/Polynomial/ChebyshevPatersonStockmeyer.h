#ifndef LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_
#define LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
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

// A C++ port of Lattigo's `bignum.OptimalSplit` function. This function
// determines the optimal split point for the Paterson-Stockmeyer algorithm by
// calculating the number of multiplications for two possible split points and
// choosing the better one.
//
// Warning: this function expresses a particular cost model counting
// multiplications when choosing two splits of the baby-step giant-step
// algorithm. However, it's not obviously clear how the calculations below
// correspond to counting multiplications, and the original author communicated
// to me (j2kun) that he doesn't remember exactly what he did here. However, he
// suspects the cost model was not perfect, and while I confirmed this was more
// optimal than what HEIR was doing earlier for a variety of test polynomials,
// we would welcome contributions to clarify the existing method, understand
// where it may fall short, and improve it.
//
// Args:
//   logDegree: std::bit_width(degree) of the polynomial degree (matching
//   golang's bits.Len64)
//
// Returns:
//   The log2 of the optimal split parameter.
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
  assert(n > 0 && "n should be greater than zero");

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
    int64_t b = n - a;
    return {a, b};
  }
}

// Computes Arithmetic DAGs of x^0, x^1, ..., x^k.
// The multiplicative depth is ceil(log2(k)).
template <typename T>
std::vector<std::shared_ptr<kernel::ArithmeticDagNode<T>>> computePowers(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x, int64_t k,
    kernel::DagType coeffType) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  bool isTensorType = coeffType.type_variant.index() >= 2;
  std::vector<std::shared_ptr<NodeTy>> result;
  result.reserve(k + 1);
  result.push_back(isTensorType ? NodeTy::splat(1, coeffType)
                                : NodeTy::constantScalar(1, coeffType));
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
//   coeffType: The type for coefficient constants
//
// Returns:
//   Node representing T_n(x)
template <typename T>
std::shared_ptr<kernel::ArithmeticDagNode<T>> genChebyshevPowerRecursive(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x, int64_t n,
    std::map<int64_t, std::shared_ptr<kernel::ArithmeticDagNode<T>>>& cache,
    kernel::DagType coeffType) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  bool isTensorType = coeffType.type_variant.index() >= 2;

  // Check cache
  auto it = cache.find(n);
  if (it != cache.end()) {
    return it->second;
  }

  if (n == 0) {
    // T_0(x) = 1
    cache[0] = isTensorType ? NodeTy::splat(1, coeffType)
                            : NodeTy::constantScalar(1, coeffType);
    return cache[0];
  }

  if (n == 1) {
    // T_1(x) = x
    cache[1] = x;
    return x;
  }

  // Split the degree optimally
  auto [a, b] = splitDegree(n);

  // Compute T_n(x) = T_a(x) * T_b(x)
  auto tA = genChebyshevPowerRecursive(x, a, cache, coeffType);
  auto tB = genChebyshevPowerRecursive(x, b, cache, coeffType);
  auto tN = NodeTy::mul(tA, tB);

  // Apply Chebyshev recurrence: T_n = 2*T_a*T_b - T_c
  // where c = |a - b|
  int64_t c = std::abs(a - b);
  auto two = isTensorType ? NodeTy::splat(2, coeffType)
                          : NodeTy::constantScalar(2, coeffType);
  tN = NodeTy::mul(two, tN);

  // Compute T_n = 2*T_a*T_b - T_c
  if (c == 0) {
    // T_0 = 1, so subtract 1
    auto one = isTensorType ? NodeTy::splat(1, coeffType)
                            : NodeTy::constantScalar(1, coeffType);
    tN = NodeTy::sub(tN, one);
  } else {
    auto tC = genChebyshevPowerRecursive(x, c, cache, coeffType);
    tN = NodeTy::sub(tN, tC);
  }

  // Cache the result
  cache[n] = tN;
  return tN;
}

// Generates Chebyshev polynomials T_0(x), T_1(x), ..., T_maxDegree(x).
//
// Args:
//   x: The input node
//   maxDegree: Maximum degree to compute
//   coeffType: The type for coefficient constants
//
// Returns:
//   Map from degree to Node representing T_degree(x)
template <typename T>
std::map<int64_t, std::shared_ptr<kernel::ArithmeticDagNode<T>>>
genChebyshevPowersRecursive(std::shared_ptr<kernel::ArithmeticDagNode<T>> x,
                            int64_t maxDegree, kernel::DagType coeffType) {
  std::map<int64_t, std::shared_ptr<kernel::ArithmeticDagNode<T>>> cache;
  bool isTensorType = coeffType.type_variant.index() >= 2;

  // Generate all powers up to maxDegree
  for (int64_t i = 1; i <= maxDegree; i++) {
    genChebyshevPowerRecursive(x, i, cache, coeffType);
  }

  cache[0] = isTensorType
      ? kernel::ArithmeticDagNode<T>::splat(1, coeffType)
      : kernel::ArithmeticDagNode<T>::constantScalar(1, coeffType);
  return cache;
}

// Computes Arithmetic DAGs of T_0(x), T_1(x), ..., T_k(x), where T_i are
// Chebyshev polynomials.
// The multiplicative depth is ceil(log2(k)).
template <typename T>
std::enable_if_t<std::is_base_of<kernel::AbstractValue, T>::value,
                 std::vector<std::shared_ptr<kernel::ArithmeticDagNode<T>>>>
computeChebyshevPolynomialValues(const T& x, int64_t k,
                                 kernel::DagType coeffType) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  bool isTensorType = coeffType.type_variant.index() >= 2;
  std::vector<std::shared_ptr<NodeTy>> result;
  result.reserve(k + 1);
  auto number1 = isTensorType ? NodeTy::splat(1, coeffType)
                              : NodeTy::constantScalar(1, coeffType);
  auto number2 = isTensorType ? NodeTy::splat(2, coeffType)
                              : NodeTy::constantScalar(2, coeffType);
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
    double minCoeffThreshold, kernel::DagType coeffType) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  bool isTensorType = coeffType.type_variant.index() >= 2;
  int64_t polynomialDegree = coefficients.size() - 1;
  if (polynomialDegree < 0) {
    return nullptr;
  }

  if (polynomialDegree == 0) {
    if (std::abs(coefficients[0]) < minCoeffThreshold) {
      return nullptr;
    }
    return isTensorType ? NodeTy::splat(coefficients[0], coeffType)
                        : NodeTy::constantScalar(coefficients[0], coeffType);
  }

  // Choose k optimally using Lattigo's optimal split
  int64_t logDegree = std::bit_width(static_cast<uint64_t>(polynomialDegree));
  int64_t logSplit = optimalSplit(logDegree);
  int64_t k = 1LL << logSplit;

  // Decompose p = p_0 + p_1 T_k + p_2 T_k^2 + ... + p_l T_k^l,
  // where each p_i is a Chebyshev polynomial of degree < k.
  polynomial::ChebyshevDecomposition decomposition =
      polynomial::decompose(coefficients, k);

  // Precompute T_0(x), T_1(x), ..., T_k(x) using recursive approach.
  auto xNode = NodeTy::leaf(x);
  auto chebPolynomialValuesMap = genChebyshevPowersRecursive(xNode, k, coeffType);

  // Evaluate the baby steps and save them in a list.
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

      auto coeffNode = isTensorType ? NodeTy::splat(coeffs[j], coeffType)
                                    : NodeTy::constantScalar(coeffs[j], coeffType);
      auto termNode = NodeTy::mul(coeffNode, chebPolynomialValuesMap[j]);

      if (pol) {
        pol = NodeTy::add(pol, termNode);
      } else {
        pol = termNode;
      }
    }
    babySteps.push_back(pol);
  }

  // Combine baby steps in tree-like manner.
  //
  // Specifically, we're evaluating
  //
  //   p = p_0 + p_1 T_k + p_2 T_k^2 + ... + p_l T_k^l,
  //
  // where the p_i are the baby steps computed above, and the value of T_k(x)
  // has been computed and stored as chebPolynomialValuesMap[k].
  //
  // This loop reduces the above terms in a tree structure to minimize depth.
  // Specifically, each round combines two adjacent terms (a, b) as a + b Y,
  // and then replaces Y with Y^2 for the next round.
  //
  // For example, starting with 4 baby steps [p_0, p_1, p_2, p_3],
  //
  // The first iteration computes [p_0 + p_1 y, p_2 + p_3 y].
  //
  // The second iteration computes [(p_0 + p_1 y) + (p_2 + p_3 y) y^2]
  //                               = p_0 + p_1 y + p_2 y^2 + p_3 y^3.
  auto y = chebPolynomialValuesMap[k];
  auto yPower = y;
  while (babySteps.size() > 1) {
    std::vector<std::shared_ptr<NodeTy>> nextBabySteps;

    for (size_t i = 0; i < babySteps.size(); i += 2) {
      auto combined = babySteps[i];

      if (i + 1 < babySteps.size() && babySteps[i + 1]) {
        auto scaled = NodeTy::mul(babySteps[i + 1], yPower);
        combined = combined ? NodeTy::add(combined, scaled) : scaled;
      }

      nextBabySteps.push_back(combined);
    }

    babySteps = std::move(nextBabySteps);
    yPower = NodeTy::mul(yPower, yPower);
  }

  return babySteps[0];
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_CHEBYSHEVPATERSONSTOCKMEYER_H_
