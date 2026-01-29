#ifndef LIB_UTILS_MATHUTILS_H_
#define LIB_UTILS_MATHUTILS_H_

#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

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

// Convert an input APFloat to the given semantics
APFloat convertFloatToSemantics(APFloat value,
                                const llvm::fltSemantics& semantics);

// Find a primitive root modulo a prime q.
std::optional<APInt> findPrimitiveRoot(const APInt& q);

// Find a primitive 2nth root of unity modulo a prime q for a given degree n.
// This requires that 2n divides q - 1.
std::optional<APInt> findPrimitive2nthRoot(const APInt& q, uint64_t n);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_MATHUTILS_H_
