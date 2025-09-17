#ifndef LIB_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
#define LIB_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_

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
    - `print-stats`: Prints statistics about the optimized circuits.
    - `mode={Boolean,LUT}`: Map gates to boolean gates or lookup table gates.
    - `use-submodules`: Extract the body of a generic op into submodules.
      Useful for large programs with generics that can be isolated. This should
      not be used when distributing generics through loops to avoid index
      arguments in the function body.
  }];

  let statistics = [
    Statistic<
      "totalCircuitSize",
      "total circuit size",
      "The total circuit size for all optimized circuits, after optimization is done."
    >,
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::comb::CombDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif  // LIB_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_TD_
