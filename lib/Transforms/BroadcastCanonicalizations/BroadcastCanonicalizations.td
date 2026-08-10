#ifndef LIB_TRANSFORMS_BROADCASTCANONICALIZATIONS_BROADCASTCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_BROADCASTCANONICALIZATIONS_BROADCASTCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def BroadcastCanonicalizations : Pass<"broadcast-canonicalizations"> {
  let summary = "This pass canonicalizes the linalg.generic broadcast operation into a linalg.broadcast.";

  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
  ];

  let description = [{
    This pass canonicalizes the linalg.generic broadcast operation into a linalg.broadcast.

    (* example filepath=tests/Transforms/broadcast_canonicalizations/broadcast.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_BROADCASTCANONICALIZATIONS_BROADCASTCANONICALIZATIONS_TD_
