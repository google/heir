#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_TD_

include "mlir/Pass/PassBase.td"

def LWEToOpenfhe : Pass<"lwe-to-openfhe"> {
  let summary = "Lower `lwe` to `openfhe` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `Openfhe` dialect.
    Currently, this also includes patterns that apply directly to `ckks` and `bgv` dialect operations.
    TODO (#1193): investigate if the need for `ckks/bgv` patterns in `--lwe-to-openfhe` is permanent.
  }];

  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_TD_
