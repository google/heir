#ifndef LIB_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_
#define LIB_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_

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

#endif  // LIB_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_
