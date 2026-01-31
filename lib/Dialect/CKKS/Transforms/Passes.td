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

def DecomposeKeySwitch : Pass<"ckks-decompose-keyswitch"> {
  let summary = "Decomposes ckks:key_switch_inner into more primitive CKKS ops";
  let description = [{
  (* example filepath=tests/Dialect/CKKS/Transforms/decompose_keyswitch.mlir *)
  }];
  let dependentDialects = ["mlir::heir::ckks::CKKSDialect", "tensor::TensorDialect"];
}

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_PASSES_TD_
