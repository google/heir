#ifndef LIB_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_TD_
#define LIB_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_TD_

include "mlir/Pass/PassBase.td"

def CGGIToTfheRust : Pass<"cggi-to-tfhe-rust"> {
  let summary = "Lower `cggi` to `tfhe_rust` dialect.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::tfhe_rust::TfheRustDialect",
  ];
}

#endif  // LIB_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_TD_
