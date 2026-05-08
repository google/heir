#ifndef LIB_TRANSFORMS_ANNOTATELEVEL_ANNOTATELEVEL_TD_
#define LIB_TRANSFORMS_ANNOTATELEVEL_ANNOTATELEVEL_TD_

include "mlir/Pass/PassBase.td"

def AnnotateLevel : Pass<"annotate-level"> {
  let summary = "Annotate level lattice values in the IR";
  let description = [{
    This pass that runs the level analysis and annotates the IR with the
    results, attaching `{mgmt.level}` attributes to each SSA value with
    an inferred value.

    If the `level-budget` is included, it gives an upper limit on the
    number of iterations the level analysis will attempt before giving
    up and reporting a default (top) value, annotated in the resulting
    IR as "invalid."

    (* example filepath=tests/Transforms/annotate_level/doctest.mlir *)
  }];
  let options = [
    Option<"levelBudget", "level-budget", "int", /*default=*/"40",
           "The level budget bound for the analysis.">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATELEVEL_ANNOTATELEVEL_TD_
