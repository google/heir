#ifndef LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_TD_
#define LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_TD_

include "mlir/Pass/PassBase.td"

def AnnotateSecretness : Pass<"annotate-secretness"> {
  let summary = "Annotate secret SSA values in the IR";
  let description = [{
    Debugging helper that runs the secretness analysis and annotates the IR with
    the results, extending the `{secret.secret}` annotation to all operation
    results that are secret.

    In addition to annotating operation results, the pass also annotates
    arguments and return types in `func.func` operations, as well as any
    terminators (e.g. `return`)

    In `verbose` mode, all results are annotated, including public ones with
    `{secret.public}`, and values for which the secretness analysis is missing
    are annotated with `{secret.missing}`, while values where the secretness
    analysis is inconclusive are annotated with `{secret.unknown}`.

    (* example filepath=tests/Transforms/annotate_secretness/doctest.mlir *)
  }];

  let options = [
    Option<"verbose", "verbose", "bool", /*default=*/"false",
           "If true, annotate secretness state all values, including public ones, and values with missing or inconclusive analysis.">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_TD_
