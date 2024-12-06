#ifndef LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
#define LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_

include "mlir/Pass/PassBase.td"

def OptimizeRelinearization : Pass <"optimize-relinearization"> {
    let summary = "Optimize placement of relinearization ops";
    let description = [{
        This pass defers relinearization ops as late as possible in the IR.
        This is more efficient in cases where multiplication operations are followed by
        additions, such as in a dot product. Because relinearization also adds error,
        deferring it can reduce the need for bootstrapping.

        In this pass, we use an integer linear program to determine the optimal
        relinearization strategy. It solves an ILP for each `func` op in the IR.

        The assumptions of this pass include:

        - All return values of functions must be linearized.
        - All ciphertext arguments to an op must have the same key basis
        - Rotation op inputs must have be linearized.

        For an ILP model specification, see the
        [docs at the HEIR website](https://heir.dev/docs/design/relinearization_ilp/).
        The model is an adaptation of the ILP described in
        [a blog post by Jeremy Kun](https://www.jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis).
    }];

    // TODO(#1032): generalize to support other schemes
    let dependentDialects = ["mlir::heir::bgv::BGVDialect"];

  let options = [
    // When using this option, one should generally write the IR to disk and
    // run `--optimize-relinearization` as the first or only pass, otherwise the
    // ILP model construction may fail due to having non-unique variable names
    Option<"useLocBasedVariableNames",
           "use-loc-based-variable-names",
           "bool",
           /*default=*/"false",
           "When true, the ILP uses op source locations in variable names, "
           "which can help debug ILP model bugs.">,
  ];
}

#endif  // LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
