#ifndef LIB_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_
#define LIB_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_

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

#endif  // LIB_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_
