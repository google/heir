#ifndef LIB_CONVERSION_BGVTOOPENFHE_BGVTOOPENFHE_TD_
#define LIB_CONVERSION_BGVTOOPENFHE_BGVTOOPENFHE_TD_

include "mlir/Pass/PassBase.td"

def BGVToOpenfhe : Pass<"bgv-to-openfhe"> {
  let summary = "Lower `bgv` to `openfhe` dialect.";

  let description = [{
    This pass lowers the `bgv` dialect to `Openfhe` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::lwe::LWEDialect",
  ];
}

#endif  // LIB_CONVERSION_BGVTOOPENFHE_BGVTOOPENFHE_TD_
