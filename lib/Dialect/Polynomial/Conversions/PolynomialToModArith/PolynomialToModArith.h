#ifndef LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_H_
#define LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_H_
