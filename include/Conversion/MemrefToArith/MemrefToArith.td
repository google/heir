#ifndef THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
#define THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_

include "mlir/Pass/PassBase.td"

def MemrefToArithPass : Pass<"memref-to-arith", "mlir::ModuleOp"> {
  let summary = "MemrefToArith lowers memrefs to their inline arithmetic values";

  let description = [{
    This pass removes MemRef creation and manipulation operations. MemRef global
     allocations are replaced by MemRefs in the referencing module. All MemRef
     aliasing operations are also folded and copies are lowered to affine loads
     and stores. At this point, all MemRef usage are write and load operations,
     so the final conversion pattern removes all affine scalar accesses by
     forwarding stores to loads and then removes the unused MemRefs.

     MemRef removal is required to remove any memory allocations from the input
     model (for example, TensorFlow models contain global memory holding model
     weights) to support FHE transpilation.
  }];

  let constructor = "createLowerMemrefToArithPass()";
}

#endif  // THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
