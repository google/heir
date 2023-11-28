#ifndef INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
#define INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_

include "mlir/Pass/PassBase.td"

def YosysOptimizer : Pass<"yosys-optimizer"> {
  let summary = "Invoke Yosys to perform circuit optimization.";

  let description = [{
    This pass invokes Yosys to convert an arithmetic circuit to an optimized
    boolean circuit that uses the arith and comb dialects.

    Note that booleanization changes the function signature: multi-bit integers
    are transformed to a tensor of booleans, for example, an `i8` is converted
    to `tensor<8xi1>`.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::comb::CombDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
