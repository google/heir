#ifndef LIB_TRANSFORMS_ACTIVATIONCANONICALIZATIONS_ACTIVATIONCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_ACTIVATIONCANONICALIZATIONS_ACTIVATIONCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def ActivationCanonicalizations : Pass<"activation-canonicalizations"> {
  let summary = "Rewrites specific activation patterns to simpler ops";
  let description = [{
    This pass rewrites activation patterns, such as ReLU implemented as a
    select and compare, to simpler operations like min/max.

    (* example filepath=tests/Transforms/activation_canonicalizations/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::heir::math_ext::MathExtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_ACTIVATIONCANONICALIZATIONS_ACTIVATIONCANONICALIZATIONS_TD_
