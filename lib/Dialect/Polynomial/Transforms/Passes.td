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
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect"];
}

def MaterializeRoots : Pass<"materialize-roots"> {
  let summary = "Attempt to materialize roots for required ring types";
  let description = [{
    Operations may require a ring to have their primitive 2n-th root of unity
    to be computed, eg. poly.ntt/intt.

    This pass determines which roots are required to have a root of unity
    provided and will attempt to materialize roots from a pre-computed table.
  }];
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect"];
}

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
