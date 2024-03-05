#ifndef INCLUDE_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_
#define INCLUDE_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_

include "mlir/Pass/PassBase.td"

def CGGIToTfheRustBool : Pass<"cggi-to-tfhe-rust-bool"> {
  let summary = "Lower `cggi` to `tfhe_rust_bool` dialect.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::tfhe_rust_bool::TfheRustBoolDialect",
  ];
}

#endif  // INCLUDE_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_
