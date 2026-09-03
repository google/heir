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
    attribute.

    Some ops support alternative approximation strategies, such as `exp` on
    `[-2^k, 1.0]` using repeated squaring `(1 + x/2^k)^(2^k)` (when using
    `--math-exp-method=taylor`), which evaluates a degree-2^k polynomial with k
    multiplicative levels.

    (* example filepath=tests/Transforms/polynomial_approximation/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"useCompositeRelu", "use-composite-relu", "bool", "false",
           "Approximate ReLU (max(x,0)) with the composite-sign method "
           "(3 chained minimax Chebyshev polynomials computing x*step(x/B), "
           "B from domain_lower/upper attrs) instead of a single-polynomial "
           "fit of max(x,0). Much more accurate for deep networks but needs "
           "more multiplicative depth / bootstrapping.">,
    Option<"mathExpMethod", "math-exp-method", "mlir::heir::MathExpMethod",
           /*default=*/"mlir::heir::MathExpMethod::Auto",
           "The method used to approximate math.exp",
           [{::llvm::cl::values(
                 clEnumValN(mlir::heir::MathExpMethod::Auto,
                            "auto", "Automatically selected method (currently defaults to CF)"),
                 clEnumValN(mlir::heir::MathExpMethod::CF,
                            "cf", "Caratheodory-Fejer minimax approximation"),
                 clEnumValN(mlir::heir::MathExpMethod::Taylor,
                            "taylor", "Taylor approximation ((1 + x/2^k)^(2^k)) via repeated squaring")
           )}]>,
  ];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
