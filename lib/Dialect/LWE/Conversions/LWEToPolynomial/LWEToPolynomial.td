#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

def LWEToPolynomial : Pass<"lwe-to-polynomial"> {
  let summary = "Lower `lwe` to `polynomial` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `polynomial` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::random::RandomDialect"
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
