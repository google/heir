#ifndef LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_TD_
#define LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_TD_

include "mlir/Pass/PassBase.td"

def CompareToSignRewrite : Pass<"compare-to-sign-rewrite"> {
  let summary = "Rewrites arith.cmpi/arith.cmpf to a math_ext.sign based expression";
  let description = [{
    This pass rewrites arith.cmpi/cmpf <pred> %a, %b to some combination of add/mul and sign operations.
    TODO(#1929): provide detailed description of the expression for each predicate.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::heir::math_ext::MathExtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_TD_
