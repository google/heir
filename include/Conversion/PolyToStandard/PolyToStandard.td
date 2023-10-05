#ifndef INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
#define INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def PolyToStandard : Pass<"poly-to-standard"> {
  let summary = "Lower `poly` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `poly` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif  // INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
