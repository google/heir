#ifndef INCLUDE_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_TD_
#define INCLUDE_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_TD_

include "mlir/Pass/PassBase.td"

def ForwardStoreToLoad : Pass<"forward-store-to-load"> {
  let summary = "Forward stores to loads within a single block";
  let description = [{
    This pass is a simplified version of mem2reg and similar passes.
    It analyzes an operation, finding all basic blocks within that op
    that have memrefs whose stores can be forwarded to loads.

    Does not support complex control flow within a block, nor ops
    with arbitrary subregions.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif  // INCLUDE_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_TD_
