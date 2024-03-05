#ifndef INCLUDE_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_TD_
#define INCLUDE_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_TD_

include "mlir/Pass/PassBase.td"

def FullLoopUnroll : Pass<"full-loop-unroll"> {
  let summary = "Fully unroll all loops";
  let description = [{
  Scan the IR for affine.for loops and unroll them all.
  }];
}

#endif  // INCLUDE_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_TD_
