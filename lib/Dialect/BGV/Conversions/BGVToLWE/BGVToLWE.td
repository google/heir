#ifndef LIB_DIALECT_BGV_CONVERSIONS_BGVTOLWE_BGVTOLWE_TD_
#define LIB_DIALECT_BGV_CONVERSIONS_BGVTOLWE_BGVTOLWE_TD_

include "mlir/Pass/PassBase.td"

def BGVToLWE : Pass<"bgv-to-lwe"> {
  let summary = "Lower `bgv` to `lwe` dialect.";

  let description = [{
    This pass lowers the `bgv` dialect to `lwe` dialect.
    Note that some scheme specific ops (e.g., modswitch) that
    have no direct analogue in the `lwe` dialect are left unchanged.
    TODO (#1193): support both "common" and "full" lwe lowering
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_BGV_CONVERSIONS_BGVTOLWE_BGVTOLWE_TD_
