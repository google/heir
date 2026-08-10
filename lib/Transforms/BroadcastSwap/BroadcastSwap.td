#ifndef LIB_TRANSFORMS_BROADCASTSWAP_BROADCASTSWAP_TD_
#define LIB_TRANSFORMS_BROADCASTSWAP_BROADCASTSWAP_TD_

include "mlir/Pass/PassBase.td"

def BroadcastSwap : Pass<"broadcast-swap"> {
  let summary = "This pass optimizes linalg.broadcast operations.";

  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
  ];

  let description = [{
    This pass detects patterns of the form: linalg.reduce then element-wise operations then linalg.broadcast. It then swaps the linalg.broadcast with the element-wise operations. This makes the linalg.broadcast a NoOp, which reduces the rotation count.

    (* example filepath=tests/Transforms/broadcast_swap/broadcast_swap.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_BROADCASTSWAP_BROADCASTSWAP_TD_
