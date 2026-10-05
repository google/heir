#ifndef LIB_UTILS_POLYNOMIAL_GOLDSCHMIDTLOOP_H_
#define LIB_UTILS_POLYNOMIAL_GOLDSCHMIDTLOOP_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/Polynomial.h"

namespace mlir {
namespace heir {
namespace polynomial {

// Approx 1/x on [lower, upper] with 0 < lower < upper is split into
//  1. An initial approx y_0 = 1/x given by a Chebyshev polynomia
//  2. A Goldschmidt loop. With e_0 = 1 - x * y_0, each iteration gives
//
//       y_{i+1} = y_i * (1 + e_i)
//       e_{i+1} = e_i * e_i
//
// Caller can evaluate the Chebyshev polynomial however it likes
// and then feed the result into the loop DAG.

struct GoldschmidtLevelSplit {
  int64_t chebyshevDegree;
  int64_t numIterations;
};

GoldschmidtLevelSplit splitGoldschmidtLevels(int64_t levels);

// Returns the Chebyshev polynomial approximating 1/x on [lower, upper]
// Requires 0 < lower < upper.
ChebyshevPolynomial goldschmidtInitialApproximation(double lower, double upper,
                                                    int64_t degree);

// Constructs an ArithmeticDag for the Goldschmidt loop.
template <typename T>
std::shared_ptr<kernel::ArithmeticDagNode<T>> goldschmidtLoop(
    std::shared_ptr<kernel::ArithmeticDagNode<T>> x,
    std::shared_ptr<kernel::ArithmeticDagNode<T>> y0, int64_t numIterations,
    kernel::DagType coeffType) {
  using NodeTy = kernel::ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;
  bool isTensorType = coeffType.type_variant.index() >= 2;
  std::vector<std::shared_ptr<NodeTy>> result;

  if (numIterations <= 0) return y0;

  auto one = isTensorType ? NodeTy::splat(1, coeffType)
                          : NodeTy::constantScalar(1, coeffType);

  auto e0 = NodeTy::sub(one, NodeTy::mul(x, y0));
  NodePtr y = y0;
  NodePtr e = e0;

  if (numIterations > 1) {  // Don't do last iter
    auto loopNode = NodeTy::loop(
        {y0, e0}, {coeffType, coeffType}, 0,
        static_cast<int32_t>(numIterations - 1), 1,
        [&](NodePtr i, const std::vector<NodePtr>& iterArgs) {
          auto yi = iterArgs[0];
          auto ei = iterArgs[1];
          return NodeTy::yield(
              {NodeTy::mul(yi, NodeTy::add(one, ei)), NodeTy::mul(ei, ei)});
        });

    y = NodeTy::resultAt(loopNode, 0);
    e = NodeTy::resultAt(loopNode, 1);
  }

  return NodeTy::mul(y, NodeTy::add(one, e));  // Last iter
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_GOLDSCHMIDTLOOP_H_
