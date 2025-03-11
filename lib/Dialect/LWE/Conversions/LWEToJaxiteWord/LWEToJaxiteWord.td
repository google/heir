#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_TD_

include "mlir/Pass/PassBase.td"

def LWEToJaxiteWord : Pass<"lwe-to-jaxiteword"> {
  let summary = "Lower `lwe` to `jaxiteword` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `JaxiteWord` dialect.
    Currently, this also includes patterns that apply directly to `ckks` and `bgv` dialect operations.
    TODO (#1193): investigate if the need for `ckks/bgv` patterns in `--lwe-to-jaxiteword` is permanent.
  }];

  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::jaxiteword::JaxiteWordDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_TD_
