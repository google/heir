#ifndef LIB_TRANSFORMS_REDUCTIONCANONICALIZATIONS_REDUCTIONCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_REDUCTIONCANONICALIZATIONS_REDUCTIONCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def ReductionCanonicalizations : Pass<"reduction-canonicalizations"> {
  let summary = "This pass canonicalizes the linalg.generic reduction operation into a linalg.reduce.";

  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
  ];

  let description = [{
    This pass canonicalizes the linalg.generic reduction operation into a linalg.reduce.

    (* example filepath=tests/Transforms/reduction_canonicalizations/torch_dot.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_REDUCTIONCANONICALIZATIONS_REDUCTIONCANONICALIZATIONS_TD_
