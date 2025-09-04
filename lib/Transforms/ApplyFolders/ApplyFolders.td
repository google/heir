#ifndef LIB_TRANSFORMS_APPLYFOLDERS_APPLYFOLDERS_TD_
#define LIB_TRANSFORMS_APPLYFOLDERS_APPLYFOLDERS_TD_

include "mlir/Pass/PassBase.td"

def ApplyFolders : Pass<"apply-folders"> {
  let summary = "Apply all folding patterns from canonicalize";
  let description = [{
    This pass applies all registered folding patterns greedily to the input IR.
    This is useful when running a full canonicalize is too slow, but applying
    folders before canonicalize is sufficient to simplify the IR for later
    passes, or even sufficient to then subsequently run a full canonicalize
    pass.

    This is used to prepare an IR for `insert-rotate` after fully unrolling
    loops.
  }];
}

#endif  // LIB_TRANSFORMS_APPLYFOLDERS_APPLYFOLDERS_TD_
