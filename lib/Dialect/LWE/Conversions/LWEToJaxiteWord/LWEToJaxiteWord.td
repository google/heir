#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITWORD_LWETOJAXITWORD_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITWORD_LWETOJAXITWORD_TD_

include "mlir/Pass/PassBase.td"

def LWEToJaxiteWord : Pass<"lwe-to-jaxiteword"> {
  let summary = "Lower `lwe` to `jaxiteword` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `jaxiteword` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::jaxiteword::JaxiteWordDialect",
    "mlir::arith::ArithDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITWORD_LWETOJAXITWORD_TD_
