#include "lib/Utils/MathUtils.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

// https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
double erfinv(double a) {
  double p, r, t;
  t = fma(a, 0.0 - a, 1.0);
  t = log(t);
  if (fabs(t) > 6.125) {            // maximum ulp error = 2.35793
    p = 3.03697567e-10;             //  0x1.4deb44p-32
    p = fma(p, t, 2.93243101e-8);   //  0x1.f7c9aep-26
    p = fma(p, t, 1.22150334e-6);   //  0x1.47e512p-20
    p = fma(p, t, 2.84108955e-5);   //  0x1.dca7dep-16
    p = fma(p, t, 3.93552968e-4);   //  0x1.9cab92p-12
    p = fma(p, t, 3.02698812e-3);   //  0x1.8cc0dep-9
    p = fma(p, t, 4.83185798e-3);   //  0x1.3ca920p-8
    p = fma(p, t, -2.64646143e-1);  // -0x1.0eff66p-2
    p = fma(p, t, 8.40016484e-1);   //  0x1.ae16a4p-1
  } else {                          // maximum ulp error = 2.35002
    p = 5.43877832e-9;              //  0x1.75c000p-28
    p = fma(p, t, 1.43285448e-7);   //  0x1.33b402p-23
    p = fma(p, t, 1.22774793e-6);   //  0x1.499232p-20
    p = fma(p, t, 1.12963626e-7);   //  0x1.e52cd2p-24
    p = fma(p, t, -5.61530760e-5);  // -0x1.d70bd0p-15
    p = fma(p, t, -1.47697632e-4);  // -0x1.35be90p-13
    p = fma(p, t, 2.31468678e-3);   //  0x1.2f6400p-9
    p = fma(p, t, 1.15392581e-2);   //  0x1.7a1e50p-7
    p = fma(p, t, -2.32015476e-1);  // -0x1.db2aeep-3
    p = fma(p, t, 8.86226892e-1);   //  0x1.c5bf88p-1
  }
  r = a * p;
  return r;
}

APFloat convertFloatToSemantics(APFloat value,
                                const llvm::fltSemantics& semantics) {
  if (&value.getSemantics() == &semantics) {
    return value;
  }
  bool losesInfo = false;
  APFloat converted = value;
  converted.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
  return converted;
}

std::optional<APInt> findPrimitiveRoot(const APInt& q) {
  if (q.ule(1)) return std::nullopt;
  if (!q[0]) {
    if (q == 2) return APInt(q.getBitWidth(), 1);
    return std::nullopt;
  }

  APInt phi = q - 1;
  std::vector<APInt> factors = factorize(phi);

  for (uint64_t g = 2; q.ugt(g); ++g) {
    APInt g_ap(q.getBitWidth(), g);
    bool is_primitive = true;
    for (const auto& p : factors) {
      if (modularExponentiation(g_ap, phi.udiv(p), q).isOne()) {
        is_primitive = false;
        break;
      }
    }
    if (is_primitive) return g_ap;
  }
  return std::nullopt;
}

/// Find a primitive 2nth root of unity modulo a prime q for a given degree n.
///
/// This implementation is a port of the primitive_2nth_root logic found in
/// sympy/ntheory/residue_ntheory.py. While SymPy provides a generalized
/// nthroot_mod for composite moduli and arbitrary n, this C++ implementation
/// is specialized for the case where q is prime and 2n divides q - 1, which
/// is the standard requirement for Number Theoretic Transforms (NTTs).
///
/// The algorithm differs from the SymPy script's use of nthroot_mod by directly
/// computing the 2n-th root from a primitive root g of q. Specifically, it
/// calculates r = g^((q-1)/2n) mod q. Since g has order q-1, the resulting
/// r is guaranteed to be a primitive 2n-th root of unity.
std::optional<APInt> findPrimitive2nthRoot(const APInt& q, uint64_t n) {
  uint64_t two_n = 2 * n;
  APInt two_n_ap(q.getBitWidth(), two_n);
  if (!q.urem(two_n_ap).isOne()) {
    // 2n must divide q - 1, which means q % 2n == 1
    return std::nullopt;
  }

  std::optional<APInt> g = findPrimitiveRoot(q);
  if (!g) return std::nullopt;

  APInt exponent = (q - 1).udiv(two_n_ap);
  return modularExponentiation(*g, exponent, q);
}

}  // namespace heir
}  // namespace mlir
