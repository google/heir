#ifndef LIB_UTILS_POLYNOMIAL_HORNER_H_
#define LIB_UTILS_POLYNOMIAL_HORNER_H_

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
    const std::map<int64_t, double>& coefficients) {
  using NodeTy = kernel::ArithmeticDagNode<T>;

  if (coefficients.empty()) {
    return NodeTy::constantScalar(0.0);
  }

  // Filter coefficients and find the highest degree
  std::map<int64_t, double> coeffMap;
  for (const auto& [degree, coeff] : coefficients) {
    coeffMap[degree] = coeff;
  }

  // Start with the coefficient of the highest degree term
  int64_t maxDegree = coeffMap.rbegin()->first;
  auto result = NodeTy::constantScalar(coeffMap[maxDegree]);

  // Apply Horner's method
  for (int64_t i = maxDegree - 1; i >= 0; i--) {
    result = NodeTy::mul(result, x);

    if (coeffMap.count(i)) {
      auto coeffNode = NodeTy::constantScalar(coeffMap.at(i));
      result = NodeTy::add(result, coeffNode);
    }
  }

  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_HORNER_H_
