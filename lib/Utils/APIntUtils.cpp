#include "lib/Utils/APIntUtils.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// Cloned after upstream removal in
/// https://github.com/llvm/llvm-project/pull/87644
///
/// Computes the multiplicative inverse of this APInt for a given modulo. The
/// iterative extended Euclidean algorithm is used to solve for this value,
/// however we simplify it to speed up calculating only the inverse, and take
/// advantage of div+rem calculations. We also use some tricks to avoid copying
/// (potentially large) APInts around.
/// WARNING: a value of '0' may be returned,
///          signifying that no multiplicative inverse exists!
APInt multiplicativeInverse(const APInt& x, const APInt& modulo) {
  assert(x.ult(modulo) && "This APInt must be smaller than the modulo");
  // Using the properties listed at the following web page (accessed 06/21/08):
  //   http://www.numbertheory.org/php/euclid.html
  // (especially the properties numbered 3, 4 and 9) it can be proved that
  // BitWidth bits suffice for all the computations in the algorithm implemented
  // below. More precisely, this number of bits suffice if the multiplicative
  // inverse exists, but may not suffice for the general extended Euclidean
  // algorithm.

  auto BitWidth = x.getBitWidth();
  APInt r[2] = {modulo, x};
  APInt t[2] = {APInt(BitWidth, 0), APInt(BitWidth, 1)};
  APInt q(BitWidth, 0);

  unsigned i;
  for (i = 0; r[i ^ 1] != 0; i ^= 1) {
    // An overview of the math without the confusing bit-flipping:
    // q = r[i-2] / r[i-1]
    // r[i] = r[i-2] % r[i-1]
    // t[i] = t[i-2] - t[i-1] * q
    x.udivrem(r[i], r[i ^ 1], q, r[i]);
    t[i] -= t[i ^ 1] * q;
  }

  // If this APInt and the modulo are not coprime, there is no multiplicative
  // inverse, so return 0. We check this by looking at the next-to-last
  // remainder, which is the gcd(*this,modulo) as calculated by the Euclidean
  // algorithm.
  if (r[i] != 1) return APInt(BitWidth, 0);

  // The next-to-last t is the multiplicative inverse.  However, we are
  // interested in a positive inverse. Calculate a positive one from a negative
  // one if necessary. A simple addition of the modulo suffices because
  // abs(t[i]) is known to be less than *this/2 (see the link above).
  if (t[i].isNegative()) t[i] += modulo;

  return std::move(t[i]);
}

APInt modularExponentiation(const APInt& base, const APInt& exponent,
                            const APInt& modulus) {
  APInt res(modulus.getBitWidth(), 1);
  APInt b = base.urem(modulus);
  APInt e = exponent;

  while (e.ugt(0)) {
    if (e[0]) {
      res = (res * b).urem(modulus);
    }
    b = (b * b).urem(modulus);
    e = e.lshr(1);
  }
  return res;
}

bool isPrime(const APInt& n) {
  if (n.ult(2)) return false;
  if (n.ult(4)) return true;
  if (!n[0]) return false;

  // Miller-Rabin primality test
  APInt d = n - 1;
  unsigned s = d.countTrailingZeros();
  d = d.lshr(s);

  // Bases to test.
  // Using the first 12 prime bases makes the test deterministic for all
  // 64-bit integers. See https://oeis.org/A014233.
  // We use 20 bases to further reduce the probability of error for
  // arbitrary-precision integers.
  std::vector<uint64_t> bases = {2,  3,  5,  7,  11, 13, 17, 19, 23, 29,
                                 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
  for (uint64_t a : bases) {
    if (n.ule(a)) break;
    APInt x = modularExponentiation(APInt(n.getBitWidth(), a), d, n);
    if (x.isOne() || x == n - 1) continue;
    bool composite = true;
    for (unsigned r = 1; r < s; ++r) {
      x = (x * x).urem(n);
      if (x == n - 1) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

std::vector<APInt> factorize(APInt n) {
  std::vector<APInt> factors;
  if (n.ult(2)) return factors;

  APInt d(n.getBitWidth(), 2);
  while ((d * d).ule(n)) {
    if (n.urem(d).isZero()) {
      factors.push_back(d);
      while (n.urem(d).isZero()) {
        n = n.udiv(d);
      }
    }
    ++d;
  }
  if (n.ugt(1)) {
    factors.push_back(n);
  }
  return factors;
}

}  // namespace heir
}  // namespace mlir
