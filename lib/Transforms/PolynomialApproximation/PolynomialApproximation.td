#ifndef LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
#define LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_

include "mlir/Pass/PassBase.td"

def PolynomialApproximation : Pass<"polynomial-approximation"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [

  ];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
