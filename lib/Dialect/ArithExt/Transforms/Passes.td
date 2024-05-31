#ifndef LIB_DIALECT_ARITHEXT_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_ARITHEXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def RemToArithExt : Pass<"convert-rem-to-arith-ext"> {
  let summary = "Rewrites `arith.remui` patterns to their arith_ext equivalents";
  let description = [{
    Applies a rewrite pattern to convert the following arith patterns to their
    equivalent using the arith_ext operations.
  }];
  let dependentDialects = ["mlir::heir::arith_ext::ArithExtDialect"];
}

#endif  // LIB_DIALECT_ARITHEXT_TRANSFORMS_PASSES_TD_
