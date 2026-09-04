#ifndef LIB_TRANSFORMS_PREPAREFORLAYOUTPROPAGATION_PREPAREFORLAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_PREPAREFORLAYOUTPROPAGATION_PREPAREFORLAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def PrepareForLayoutPropagation : Pass<"prepare-for-layout-propagation", "ModuleOp"> {
  let summary = "Normalize surviving linalg/arith forms into shapes layout propagation supports.";
  let description = [{
    Runs at the end of linalg preprocessing, after general
    canonicalization (whose own patterns, e.g. the average-pool rewrites,
    must see the original forms first):

    - Raises a broadcast-shaped `linalg.generic` (all-parallel, one input
      read through a strictly increasing projection map and yielded
      unchanged, identity output map) to the named `linalg.broadcast`.
      torch-mlir emits this form e.g. for LayerNorm's mean/variance
      broadcasts; layout propagation only understands the named op.

    - Rewrites `x / splat-constant` into `x * splat(1/constant)`. Division
      by a compile-time constant is the only float division an FHE backend
      can evaluate, and only as a multiplication; torch exports emit it
      e.g. for a LayerNorm mean (sum / N), where the linear-op div fusion
      does not apply.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

#endif  // LIB_TRANSFORMS_PREPAREFORLAYOUTPROPAGATION_PREPAREFORLAYOUTPROPAGATION_TD_
