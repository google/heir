#ifndef INCLUDE_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_TD_
#define INCLUDE_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_TD_

include "mlir/Pass/PassBase.td"

def CGGIToTfheRust : Pass<"cggi-to-tfhe-rust"> {
  let summary = "Lower `cggi` to `tfhe_rust` dialect.";
  let dependentDialects = [
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::tfhe_rust::TfheRustDialect",
  ];
}

#endif  // INCLUDE_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_TD_
