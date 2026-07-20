#ifndef LIB_UTILS_APPROXIMATION_TAYLOR_H_
#define LIB_UTILS_APPROXIMATION_TAYLOR_H_

#include <cstdint>

namespace mlir {
namespace heir {
namespace approximation {

/// Evaluates the Taylor exponential approximation (1 + x / 2^k)^(2^k)
/// via repeated squaring for a given input x and parameter k.
inline double expTaylorApproximation(double x, int64_t k = 7) {
  double scale = 1.0 / static_cast<double>(1ULL << k);
  double val = 1.0 + x * scale;
  for (int64_t i = 0; i < k; ++i) {
    val = val * val;
  }
  return val;
}

}  // namespace approximation
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APPROXIMATION_TAYLOR_H_
