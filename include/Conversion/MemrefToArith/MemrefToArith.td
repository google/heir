#ifndef THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
#define THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_

include "mlir/Pass/PassBase.td"

def MemrefGlobalReplacePass : Pass<"memref-global-replace", "mlir::ModuleOp"> {
  let summary = "MemrefGlobalReplacePass forwards global memrefs accessors "
  "to arithmetic values";

  let description = [{
    This pass forwards constant global MemRef values to referencing affine
    loads. This pass requires that the MemRef global values are initialized as
    constants and that the affine load access indices are constants (i.e. not
    variadic). Unroll affine loops prior to running this pass.

     MemRef removal is required to remove any memory allocations from the input
     model (for example, TensorFlow models contain global memory holding model
     weights) to support FHE transpilation.
  }];

  let constructor = "createMemrefGlobalReplacePass()";
}

def ExpandCopyPass : Pass<"expand-copy", "mlir::ModuleOp"> {
  let summary = "Expands memref.copy ops to explicit affine loads and stores";

  let description = [{
    This pass removes memref copy operations by expanding it to affine loads and
    stores. This pass introduces affine loops over the dimensions of the MemRef,
    so must be run prior to any affine loop unrolling in a pipeline.
  }];

  let constructor = "createExpandCopyPass()";
}

def UnrollAndForwardPass : Pass<"unroll-and-forward", "mlir::ModuleOp"> {
  let summary = "Loop unrolls and forwards stores to loads.";

  let description = [{
    This pass proceses the first function in a given module, and, starting from
    the first loop, iteratively does the following:

    1. Fully unroll the loop.
    2. Scan for load ops. For each load op with a statically-inferrable access
    index:
      1. Backtrack to the original memref alloc
      2. Find all store ops at the corresponding index (possibly transitively
      through renames/subviews of the underlying alloc).
      3. Find the last store that occurs and forward it to the load.
      4. If the original memref is an input memref, then forward through any
      renames to make the target load load directly from the argument memref
      (instead of any subviews, say)
    3. Apply the same logic to any remaining loads not inside any for loop.

    This pass requires that tensors are lowered to memref, and only supports
    affine loops with affine.load/store ops.

    Memrefs that result from memref.get_global ops are excluded from
    forwarding, even if they are loaded with a static index, and are instead
    handled by memref-global-replace, which should be run after this pass.
  }];

  let constructor = "createUnrollAndForwardStoresPass()";
}

#endif  // THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
