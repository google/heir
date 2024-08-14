#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PolyMulToNTT : Pass<"convert-polynomial-mul-to-ntt"> {
  let summary = "Rewrites polynomial operations to their NTT equivalents";
  let description = [{
    Applies a rewrite pattern to convert polynomial multiplication to the
    equivalent using the number-theoretic transforms (NTT) when possible.

    Polynomial multiplication can be rewritten as polynomial.NTT
    on each operand, followed by modulo elementwise multiplication of the
    point-value representation and then the inverse-NTT back to coefficient
    representation.
  }];
  let dependentDialects = ["mlir::polynomial::PolynomialDialect", "heir::mod_arith::ModArithDialect"];
}

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
