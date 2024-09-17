#ifndef LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def LinalgCanonicalizations : Pass<"linalg-canonicalizations"> {
  let summary = "This pass canonicalizes the linalg.transpose operation of a constant into a transposed constant.";

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let description = [{
    This pass canonicalizes the linalg.transpose operation of a constant into a
    transposed constant.
  }];
}

#endif  // LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_
