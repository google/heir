#ifndef LIB_UTILS_POLYNOMIAL_PATERSONSTOCKMEYER_H_
#define LIB_UTILS_POLYNOMIAL_PATERSONSTOCKMEYER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"

namespace mlir {
namespace heir {
namespace polynomial {

// Creates Arithmetic DAG for evaluating a monomial polynomial using
// the Paterson-Stockmeyer method.
template <typename T>
std::shared_ptr<kernel::ArithmeticDagNode<T>>
patersonStockmeyerMonomialPolynomialEvaluation(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x,
    const std::map<int64_t, double>& coefficients) {
  using NodeTy = kernel::ArithmeticDagNode<T>;

  if (coefficients.empty()) {
    return NodeTy::constantScalar(0.0);
  }

  // Filter coefficients
  std::map<int64_t, double> coeffMap;
  for (const auto& [degree, coeff] : coefficients) {
    coeffMap[degree] = coeff;
  }

  int64_t maxDegree = coeffMap.rbegin()->first;

  // Choose k optimally - sqrt of maxDegree is typically a good choice
  int64_t k = std::max(static_cast<int64_t>(std::ceil(std::sqrt(maxDegree))),
                       static_cast<int64_t>(1));

  // Precompute x^1, x^2, ..., x^k
  std::vector<std::shared_ptr<NodeTy>> xPowers = computePowers(x, k);

  // Number of chunks we'll need
  int64_t m =
      static_cast<int64_t>(std::ceil(static_cast<double>(maxDegree + 1) / k));
  std::vector<std::shared_ptr<NodeTy>> chunkValues;

  for (int64_t i = 0; i < m; i++) {
    int64_t highestDegreeInChunk = std::min((i + 1) * k - 1, maxDegree);
    int64_t lowestDegreeInChunk = i * k;

    std::shared_ptr<NodeTy> chunkValue = nullptr;

    for (int64_t j = lowestDegreeInChunk; j <= highestDegreeInChunk; j++) {
      if (coeffMap.count(j)) {
        int64_t powerIndex = j - lowestDegreeInChunk;
        auto coeff = NodeTy::constantScalar(coeffMap[j]);

        std::shared_ptr<NodeTy> term;
        if (powerIndex == 0) {
          term = coeff;
        } else {
          term = NodeTy::mul(coeff, xPowers[powerIndex]);
        }

        if (!chunkValue) {
          chunkValue = term;
        } else {
          chunkValue = NodeTy::add(chunkValue, term);
        }
      }
    }

    if (!chunkValue) {
      chunkValue = NodeTy::constantScalar(0.0);
    }
    chunkValues.push_back(chunkValue);
  }

  // Combine the chunks using Horner's method on the chunks
  std::shared_ptr<NodeTy> result = chunkValues.back();
  for (int64_t i = m - 2; i >= 0; i--) {
    result = NodeTy::mul(result, xPowers[k]);
    result = NodeTy::add(result, chunkValues[i]);
  }

  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_PATERSONSTOCKMEYER_H_
