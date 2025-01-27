#ifndef LIB_DIALECT_LINALG_CONVERSIONS_LINALGTOTENSOREXT_LINALGTOTENSOREXT_TD_
#define LIB_DIALECT_LINALG_CONVERSIONS_LINALGTOTENSOREXT_LINALGTOTENSOREXT_TD_

include "mlir/Pass/PassBase.td"

def LinalgToTensorExt : Pass<"linalg-to-tensor-ext"> {
  let summary = "Lower `linalg.matmul` to arith and tensor_ext dialects.";

  let description = [{
    This pass lowers the `linalg.matmul` to a mixture of affine, tensor, and
    via the Halevi-Shoup and squat matrix multiplication algorithms.

    We assume that the input and output values are replicated. This makes
    aligning the matrix multiplications easier (though not necessarily optimal).
    For example, when multiplying a 1x4 vector with a 4x2 matrix, the bias and output
    will be a 1x2 vector. However, due to requiring tensor sizes to match, and
    assuming replication, the matrix will be expanded to a 4x4 matrix and output
    to a 1x4 vector (where the output is replicated twice).

    For now, the tilingSize is a command line parameter that determines the
    maximum secret vector size used in the Halevi-Shoup and squat matrix
    multiplication algorithms. It can be specified via --linalg-to-tensor-ext=tiling-size=16.
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
  let options = [
    Option<"tilingSize", "tiling-size", "int", "16", "tiling size of the halevi-shoup and squat packing matrix multiplication algorithms">
  ];
}

#endif  // LIB_DIALECT_LINALG_CONVERSIONS_LINALGTOTENSOREXT_LINALGTOTENSOREXT_TD_
