#ifndef LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_
#define LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_

include "mlir/Pass/PassBase.td"

def CKKSToLWE : Pass<"ckks-to-lwe"> {
  let summary = "Lower `ckks` to `lwe` dialect.";

  let description = [{
    This pass lowers the `ckks` dialect to `lwe` dialect.
    Note that some scheme specific ops (e.g., rescale) that
    have no direct analogue in the `lwe` dialect are left unchanged.
    TODO (#1193): support both "common" and "full" lwe lowering
  }];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_
