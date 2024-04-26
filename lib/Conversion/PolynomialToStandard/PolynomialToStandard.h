#ifndef LIB_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_H_
#define LIB_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL
#include "lib/Conversion/PolynomialToStandard/PolynomialToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/PolynomialToStandard/PolynomialToStandard.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_H_
