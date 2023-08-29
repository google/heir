#ifndef HEIR_INCLUDE_CONVERSION_BGVTOPOLY_BGVTOPOLY_TD_
#define HEIR_INCLUDE_CONVERSION_BGVTOPOLY_BGVTOPOLY_TD_

include "mlir/Pass/PassBase.td"

def BGVToPoly : Pass<"bgv-to-poly"> {
  let summary = "Lower `bgv` to `poly` dialect.";

  let description = [{
    This pass lowers the `bgv` dialect to `poly` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // HEIR_INCLUDE_CONVERSION_BGVTOPOLY_BGVTOPOLY_TD_
