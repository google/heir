#ifndef LIB_CONVERSION_CGGITOOPENFHE_CGGITOOPENFHE_TD_
#define LIB_CONVERSION_CGGITOOPENFHE_CGGITOOPENFHE_TD_

include "mlir/Pass/PassBase.td"

def CGGIToOpenfhe : Pass<"cggi-to-openfhe"> {
  let summary = "Lower `cggi` to `openfhe` dialect.";
  let dependentDialects = [
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::openfhe::OpenfheDialect",
  ];
}

#endif  // LIB_CONVERSION_CGGITOOPENFHE_CGGITOOPENFHE_TD_