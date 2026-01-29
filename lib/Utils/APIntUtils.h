#ifndef LIB_UTILS_APINTUTILS_H_
#define LIB_UTILS_APINTUTILS_H_

#include <vector>

#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

APInt multiplicativeInverse(const APInt& x, const APInt& modulo);

APInt modularExponentiation(const APInt& base, const APInt& exponent,
                            const APInt& modulus);

bool isPrime(const APInt& n);

/// Returns the prime factors of n, without multiplicity. If n < 2, returns an
/// empty vector.
std::vector<APInt> factorize(APInt n);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APINTUTILS_H_
