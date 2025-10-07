#ifndef LIB_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TD_
#define LIB_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TD_

include "mlir/Pass/PassBase.td"

def TensorExtToTensor : Pass<"tensor-ext-to-tensor"> {
  let summary = "Lower `tensor_ext` to `tensor` dialect.";
  let description = [{
    This pass lowers the `tensor_ext` dialect to the `tensor` dialect.

    This pass is intended to be used for testing purpose where the
    secret arithmetic IR containing `tensor_ext` dialect is lowered
    to the IR containing `tensor` dialect, which could be further
    lowered to the LLVM dialect.

    (* example filepath=tests/Dialect/TensorExt/Conversions/tensor_ext_to_tensor/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_TD_
