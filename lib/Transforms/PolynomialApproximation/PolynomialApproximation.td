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

    The following ops in the `arith` dialect are also supported:

    - `maxf`
    - `maxnumf`
    - `minf`
    - `minnumf`

    These ops are replaced with `polynomial.eval` ops with a static polynomial
    attribute.

    Examples:

    ```mlir
    %0 = math.exp %x {
          degree = 3 : i32,
          domain_lower = -1.0 : f64,
          domain_upper = 1.0 : f64
      } : f32
    ```

    is converted to

    ```mlir
    #ring_f64_ = #polynomial.ring<coefficientType = f64>
    !poly = !polynomial.polynomial<ring = #ring_f64_>
    %0 = polynomial.eval
           #polynomial<typed_float_polynomial <
             0.99458116404270657
           + 0.99565537253615788x
           + 0.54297028147256321x**2
           + 0.17954582110873779x**3> : !poly>, %arg0 : f32
    ```
  }];
  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect"
  ];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
