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

#endif  // THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
