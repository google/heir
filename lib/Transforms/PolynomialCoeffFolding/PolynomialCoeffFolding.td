#ifndef LIB_TRANSFORMS_POLYNOMIALCOEFFFOLDING_POLYNOMIALCOEFFFOLDING_TD_
#define LIB_TRANSFORMS_POLYNOMIALCOEFFFOLDING_POLYNOMIALEFFFOLDING_TD_

include "mlir/Pass/PassBase.td"

def PolynomialCoeffFolding : Pass<"polynomial-coeff-folding"> {
  let summary = "Folds constants into polynomial evaluation";
  let description = [{
    This pass folds constants into `polynomial.eval` operations.

    It matches operations before and after `polynomial.eval` and updates the
    polynomial coefficients accordingly.

    (* example filepath=tests/Transforms/polynomial_coeff_folding/doctest.mlir *)
  }];

}

#endif // LIB_TRANSFORMS_POLYNOMIALCOEFFFOLDING_POLYNOMIALCOEFFFOLDING_TD_
