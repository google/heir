#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_TD_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_TD_

include "mlir/Pass/PassBase.td"

def OrionToCKKS : Pass<"orion-to-ckks"> {
  let summary = "Lower `orion` to `ckks` dialect.";
  let dependentDialects = [
    "mlir::heir::orion::OrionDialect",
    "mlir::heir::ckks::CKKSDialect",
  ];
}

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_TD_
