#ifndef LIB_CONVERSION_LINALGTOTENSOREXT_LINALGTOTENSOREXT_TD_
#define LIB_CONVERSION_LINALGTOTENSOREXT_LINALGTOTENSOREXT_TD_

include "mlir/Pass/PassBase.td"

def LinalgToTensorExt : Pass<"linalg-to-tensor-ext"> {
  let summary = "Lower `linalg.matmul` to arith and tensor_ext dialects.";

  let description = [{
    This pass lowers the `linalg.matmul` to a mixture of affine, tensor, and
    via the Halevi-Shoup and squat matrix multiplication algorithms.
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // LIB_CONVERSION_LINALGTOTENSOREXT_LINALGTOTENSOREXT_TD_
