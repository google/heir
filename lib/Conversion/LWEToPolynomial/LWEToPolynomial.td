#ifndef LIB_CONVERSION_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
#define LIB_CONVERSION_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

def LWEToPolynomial : Pass<"lwe-to-polynomial"> {
  let summary = "Lower `lwe` to `polynomial` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `polynomial` dialect.
  }];

  let dependentDialects = [
    "mlir::polynomial::PolynomialDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::random::RandomDialect"
  ];
}

#endif  // LIB_CONVERSION_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
