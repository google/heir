#ifndef LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_H_
#define LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_H_

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace to_mod_arith {

#define GEN_PASS_DECL
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

}  // namespace to_mod_arith
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_H_
