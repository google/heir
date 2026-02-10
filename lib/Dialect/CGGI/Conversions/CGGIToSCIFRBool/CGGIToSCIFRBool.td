#ifndef LIB_DIALECT_SCIFRBOOL_CONVERSIONS_CGGITOSCIFRBOOL_TD_
#define LIB_DIALECT_SCIFRBOOL_CONVERSIONS_CGGITOSCIFRBOOL_TD_

include "mlir/Pass/PassBase.td"

def CGGIToSCIFRBool : Pass<"cggi-to-scifrbool"> {
  let summary = "Lower `cggi` to `scifrbool` dialect.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::scifrbool::SCIFRBoolDialect",
  ];
}

#endif  // LIB_DIALECT_SCIFRBOOL_CONVERSIONS_CGGITOSCIFRBOOL_TD_
