#ifndef INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
#define INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_

include "mlir/Pass/PassBase.td"

def YosysOptimizer : Pass<"yosys-optimizer"> {
  let summary = "Invoke Yosys to perform circuit optimization.";

  let description = [{
    This pass invokes Yosys to convert an arithmetic circuit to an optimized
    boolean circuit that uses the arith and comb dialects.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::comb::CombDialect",
  ];
}

#endif  // INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
