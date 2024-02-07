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

    The optimizer will be applied to each `secret.generic` op containing
    arithmetic ops that can be optimized.

    Optional parameters:

    - `abc-fast`: Run the abc optimizer in "fast" mode, getting faster compile
      time at the expense of a possibly larger output circuit.
    - `unroll-factor`: Before optimizing the circuit, unroll loops by a given
      factor. If unset, this pass will not unroll any loops.
  }];
  // TODO(https://github.com/google/heir/issues/257): add option for the pass to select
  // the unroll factor automatically.

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::comb::CombDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def UnrollAndOptimizeAnalysis : Pass<"unroll-and-optimize-analysis"> {
  let summary = "Iteratively unroll and optimize an IR, printing optimization stats.";

  let description = [{
    This pass invokes the `--yosys-optimizer` pass while iteratively applying
    loop-unrolling. Along the way, it prints statistics about the optimized
    circuits, which can be used to determine an optimal loop-unrolling factor
    for a given program.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::comb::CombDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
