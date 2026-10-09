#ifndef LIB_UTILS_MATHUTILS_H_
#define LIB_UTILS_MATHUTILS_H_

#include <bit>
#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/APFloat.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"           // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project

namespace mlir {
namespace heir {

/// inverse error function
double erfinv(double a);

inline bool isPowerOfTwo(int64_t n) {
  return n > 0 && llvm::isPowerOf2_64(static_cast<uint64_t>(n));
}

// Levels consumed by Lattigo's polynomial evaluator for a Chebyshev polynomial
// of the given degree: one per level of the binary evaluation tree.
inline int lattigoChebyshevDepth(uint32_t degree) {
  return std::bit_width(static_cast<uint64_t>(degree));
}

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
