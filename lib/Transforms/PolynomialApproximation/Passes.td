#ifndef LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_PASSES_TD_
#define LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ApproximateRelu : Pass<"approximate-relu"> {
  let summary = "Approximate ReLU-like ops with a polynomial";
  let description = [{
  This pass applies a polynomial approximation to ops that
  resemble `f(x) = max(0, x)`, including:

  - `tosa.clamp` with one input zero (or the zero-point of a quantization).
  }];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_PASSES_TD_
