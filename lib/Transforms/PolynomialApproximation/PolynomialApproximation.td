#ifndef LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
#define LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_

include "mlir/Pass/PassBase.td"

def PolynomialApproximation : Pass<"polynomial-approximation"> {
  let summary = "Approximate ops by polynomials";
  let description = [{
    This pass replaces certain operations that are incompatible
    with the FHE computational model with polynomial approximations.

    The pass applies to the following ops:

    - `math.exp`

    These ops are replaced with `polynomial.eval` ops with a static
    polynomial attribute.

    Examples:

    // FIXME: add examples
  }];
  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect"
  ];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
