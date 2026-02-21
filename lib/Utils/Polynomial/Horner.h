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
    const std::map<int64_t, double>& coefficients, kernel::DagType dagType) {
  using NodeTy = kernel::ArithmeticDagNode<T>;

  bool isTensorType = dagType.type_variant.index() >= 2;

  if (coefficients.empty()) {
    if (isTensorType) {
      return NodeTy::splat(0.0, dagType);
    }
    return NodeTy::constantScalar(0.0, dagType);
  }

  // Filter coefficients and find the highest degree
  std::map<int64_t, double> coeffMap;
  for (const auto& [degree, coeff] : coefficients) {
    coeffMap[degree] = coeff;
  }

  // Start with the coefficient of the highest degree term
  int64_t maxDegree = coeffMap.rbegin()->first;
  auto result = isTensorType
                    ? NodeTy::splat(coeffMap[maxDegree], dagType)
                    : NodeTy::constantScalar(coeffMap[maxDegree], dagType);

  // Apply Horner's method
  for (int64_t i = maxDegree - 1; i >= 0; i--) {
    result = NodeTy::mul(result, x);

    if (coeffMap.count(i)) {
      auto coeffNode = isTensorType
                           ? NodeTy::splat(coeffMap.at(i), dagType)
                           : NodeTy::constantScalar(coeffMap.at(i), dagType);
      result = NodeTy::add(result, coeffNode);
    }
  }

  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_HORNER_H_
