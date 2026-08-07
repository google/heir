#ifndef LIB_TRANSFORMS_VALIDATESCALE_VALIDATESCALE_TD_
#define LIB_TRANSFORMS_VALIDATESCALE_VALIDATESCALE_TD_

include "mlir/Pass/PassBase.td"

def ValidateScale : Pass<"validate-scale"> {
  let summary = "Validate the scale correctness of annotated IR";
  let description = [{
    This pass validates the scale correctness of the annotated IR for CKKS and BGV.
  }];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::mgmt::MgmtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_VALIDATESCALE_VALIDATESCALE_TD_
