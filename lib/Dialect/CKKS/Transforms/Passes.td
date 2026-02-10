#ifndef LIB_DIALECT_CKKS_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CKKS_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DecomposeRelinearize : Pass<"ckks-decompose-relinearize"> {
  let summary = "Decomposes ckks:relinearize into ckks:keyswitchinner and addition";
  let description = [{
  (* example filepath=tests/Dialect/CKKS/Transforms/decompose_relinearize.mlir *)
  }];
  let dependentDialects = ["mlir::heir::ckks::CKKSDialect"];
}

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_PASSES_TD_
