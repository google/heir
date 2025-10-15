#ifndef LIB_TRANSFORMS_RELUCANONICALIZATIONS_RELUCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_RELUCANONICALIZATIONS_RELUCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def ReLUCanonicalizations : Pass<"relu-canonicalizations"> {
  let summary = "Rewrites comparison and selection to a min/max";
  let description = [{
    This pass rewrites selections with a constant comparison to a min/max
    operation.

    (* example filepath=tests/Transforms/relu_canonicalizations/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::heir::math_ext::MathExtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_RELUCANONICALIZATIONS_RELUCANONICALIZATIONS_TD_
