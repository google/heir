#include "lib/Utils/MathUtils.h"

#include <cmath>

#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
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

}  // namespace heir
}  // namespace mlir
