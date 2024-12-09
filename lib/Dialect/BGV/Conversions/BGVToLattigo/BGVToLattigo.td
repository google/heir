#ifndef LIB_DIALECT_BGV_CONVERSIONS_BGVTOLATTIGO_BGVTOLATTIGO_TD_
#define LIB_DIALECT_BGV_CONVERSIONS_BGVTOLATTIGO_BGVTOLATTIGO_TD_

include "mlir/Pass/PassBase.td"

def BGVToLattigo : Pass<"bgv-to-lattigo"> {
  let summary = "Lower `bgv` to `lattigo` dialect.";

  let description = [{
    This pass lowers the `bgv` dialect to `Lattigo` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::lattigo::LattigoDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::lwe::LWEDialect",
  ];
}

#endif  // LIB_DIALECT_BGV_CONVERSIONS_BGVTOLATTIGO_BGVTOLATTIGO_TD_
