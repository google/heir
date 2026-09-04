#ifndef LIB_TRANSFORMS_GATHERZEROENCRYPTIONS_PASSES_TD_
#define LIB_TRANSFORMS_GATHERZEROENCRYPTIONS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def GatherZeroEncryptions : Pass<"gather-zero-encryptions", "mlir::ModuleOp"> {
  let summary = "Gather zero encryption arguments into a single memref/array";

  let description = [{
    This pass gathers zero encryption arguments created for trivial encryptions
    into a single memref argument and combines the corresponding client encryption
    helper functions into a single helper function producing the memref.

    (* example filepath=tests/Transforms/gather_zero_encryptions/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_TRANSFORMS_GATHERZEROENCRYPTIONS_PASSES_TD_
