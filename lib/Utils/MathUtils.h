#ifndef LIB_UTILS_MATHUTILS_H_
#define LIB_UTILS_MATHUTILS_H_

#include <cstdint>

namespace mlir {
namespace heir {

/// inverse error function
double erfinv(double a);

inline uint64_t nextPowerOfTwo(uint64_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v++;

  return v;
}

inline bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_MATHUTILS_H_
