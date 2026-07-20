#ifndef LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
#define LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_

include "mlir/Pass/PassBase.td"

def PolynomialApproximation : Pass<"polynomial-approximation"> {
  let summary = "Approximate ops by polynomials";
  let description = [{
    This pass replaces certain operations that are incompatible
    with the FHE computational model with polynomial approximations.

    The pass applies to the following ops in the `math` dialect. When the
    op is binary, the pass applies when one op is the result of an
    `arith.constant` which is scalar-valued or a splatted tensor.

    - `absf`
    - `acos`
    - `acosh`
    - `asin`
    - `asinh`
    - `atan2`
    - `atan`
    - `atanh`
    - `cbrt`
    - `ceil`
    - `copysign`
    - `cos`
    - `cosh`
    - `erf`
    - `erfc`
    - `exp2`
    - `exp`
    - `expm1`
    - `floor`
    - `fpowi`
    - `log10`
    - `log1p`
    - `log2`
    - `log`
    - `powf`
    - `round`
    - `roundeven`
    - `rsqrt`
    - `sin`
    - `sinh`
    - `sqrt`
    - `tan`
    - `tanh`
    - `trunc`

    As well as the following ops in the `math_ext` dialect:
    - `sign`

    The following ops in the `arith` dialect are also supported:

    - `maxf`
    - `maxnumf`
    - `minf`
    - `minnumf`

    These ops are replaced with `polynomial.eval` ops with a static polynomial
    attribute, with the exception of `exp` operations with domain in `[-2^k, 1.0]`
    (default `k=7`), which are replaced using a Taylor approximation
    `e^x = (1 + x/2^k)^(2^k)` evaluated via repeated squaring.

    (* example filepath=tests/Transforms/polynomial_approximation/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect"
  ];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
