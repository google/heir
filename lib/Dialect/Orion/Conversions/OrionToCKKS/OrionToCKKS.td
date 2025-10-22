#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_TD_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_TD_

include "mlir/Pass/PassBase.td"

def OrionToCKKS : Pass<"orion-to-ckks"> {
  let summary = "Lower `orion` to `ckks` dialect.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::orion::OrionDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::tensor::TensorDialect",
  ];

  let options = [
    Option<
      "libraryTarget",
      "library-target",
      "std::string",
      /*default=*/"\"openfhe\"",
      "Backend library to specialize for when lowering chebyshev and linear_transform ops."
    >,
  ];
}

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_TD_
