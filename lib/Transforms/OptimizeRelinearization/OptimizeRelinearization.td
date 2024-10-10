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

        In much of the literature, lazy relinearization is applied manually. See for example
        Blatt-Gusev-Polyakov-Rohloff-Vaikuntanathan 2019 https://eprint.iacr.org/2019/223
        and Lee-Lee-Kim-Kim-No-Kang 2020 https://eprint.iacr.org/2020/1549.

        In the EVA compiler https://eprint.iacr.org/2021/1505.pdf,
        relinearization is applied automatically via a heuristic, either
        immediately after each multiplication op, or "lazily," deferred as late
        as possible.

        In this pass, we use an integer linear program to determine the optimal relinearization
        strategy. This is an adaptation of the ILP described in
        https://www.jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis

        TODO(#1026): document the ILP spec
    }];

    // TODO(#1032): generalize to support other schemes
    let dependentDialects = ["mlir::heir::bgv::BGVDialect"];
}

#endif  // LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
