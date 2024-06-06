#ifndef LIB_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_TD_
#define LIB_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

def BGVToPolynomial : Pass<"bgv-to-polynomial"> {
  let summary = "Lower `bgv` to `polynomial` dialect.";

  let description = [{
    This pass lowers the `bgv` dialect to `polynomial` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::polynomial::PolynomialDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_TD_
