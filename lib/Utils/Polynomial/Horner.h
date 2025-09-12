#ifndef LIB_UTILS_POLYNOMIAL_HORNER_H_
#define LIB_UTILS_POLYNOMIAL_HORNER_H_

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>

#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace polynomial {

// Creates Arithmetic DAG for evaluating a monomial polynomial using Horner's
// method.
template <typename T>
std::shared_ptr<kernel::ArithmeticDagNode<T>>
hornerMonomialPolynomialEvaluation(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x,
    const std::map<int64_t, double>& coefficients,
    double minCoeffThreshold = 1e-12) {
  using NodeTy = kernel::ArithmeticDagNode<T>;

  if (coefficients.empty()) {
    return NodeTy::constant(0.0);
  }

  // Filter coefficients and find the highest degree
  std::map<int64_t, double> filteredCoeffs;
  for (const auto& [degree, coeff] : coefficients) {
    if (std::abs(coeff) >= minCoeffThreshold) {
      filteredCoeffs[degree] = coeff;
    }
  }

  if (filteredCoeffs.empty()) {
    return NodeTy::constant(0.0);
  }

  int64_t maxDegree = filteredCoeffs.rbegin()->first;

  // Start with the coefficient of the highest degree term
  auto result = NodeTy::constant(filteredCoeffs[maxDegree]);

  // Apply Horner's method
  for (int64_t i = maxDegree - 1; i >= 0; i--) {
    result = NodeTy::mul(result, x);

    if (filteredCoeffs.count(i)) {
      auto coeffNode = NodeTy::constant(filteredCoeffs.at(i));
      result = NodeTy::add(result, coeffNode);
    }
  }

  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_HORNER_H_
