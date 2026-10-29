#ifndef LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_TD_
#define LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_TD_

include "mlir/Pass/PassBase.td"

def CompareToSignRewrite : Pass<"compare-to-sign-rewrite"> {
  let summary = "Rewrites arith.cmpi/arith.cmpf to a math_ext.sign based expression";
  let description = [{
    This pass rewrites arith.cmpi/cmpf <pred> %a, %b to some combination of
    add/mul and sign operations. Since `math_ext.sign` returns -1, 0, or 1 for
    negative, zero, or positive inputs respectively, the strict comparisons
    are rewritten as

      a < b  ->  0.5 * (sign(b - a) + 1)
      a > b  ->  0.5 * (sign(a - b) + 1)

    TODO(#1929): support the remaining predicates (equality, <=, >=).

    (* example filepath=tests/Transforms/compare_to_sign_rewrite/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::heir::math_ext::MathExtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_TD_
