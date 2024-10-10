#ifndef LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOOPENFHE_CKKSTOOPENFHE_TD_
#define LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOOPENFHE_CKKSTOOPENFHE_TD_

include "mlir/Pass/PassBase.td"

def CKKSToOpenfhe : Pass<"ckks-to-openfhe"> {
  let summary = "Lower `ckks` to `openfhe` dialect.";

  let description = [{
    This pass lowers the `ckks` dialect to `Openfhe` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::lwe::LWEDialect",
  ];
}

#endif  // LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOOPENFHE_CKKSTOOPENFHE_TD_
