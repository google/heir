#ifndef INCLUDE_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_
#define INCLUDE_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_

include "mlir/Pass/PassBase.td"

def RemoveUnusedMemRef : Pass<"remove-unused-memref", "func::FuncOp"> {
  let summary = "Cleanup any unused memrefs";
  let description = [{
    Scan the IR for unused memrefs and remove them.

    This pass looks for locally allocated memrefs that are never used and
    deletes them. This pass can be used as a cleanup pass from other IR
    simplifications that forward stores to loads.
  }];
}

#endif  // INCLUDE_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_
