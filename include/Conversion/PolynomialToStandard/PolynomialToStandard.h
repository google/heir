#ifndef INCLUDE_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_H_
#define INCLUDE_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL
#include "include/Conversion/PolynomialToStandard/PolynomialToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/PolynomialToStandard/PolynomialToStandard.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_H_
