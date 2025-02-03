#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_TD_

include "mlir/Pass/PassBase.td"

def LWEToLattigo : Pass<"lwe-to-lattigo"> {
  let summary = "Lower `lwe` to `lattigo` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `Lattigo` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::lattigo::LattigoDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::lwe::LWEDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_TD_
