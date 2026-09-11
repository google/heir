#ifndef LIB_TRANSFORMS_LOWERPADTOIDENTITYMATMUL_LOWERPADTOIDENTITYMATMUL_TD_
#define LIB_TRANSFORMS_LOWERPADTOIDENTITYMATMUL_LOWERPADTOIDENTITYMATMUL_TD_

include "mlir/Pass/PassBase.td"

def LowerPadToIdentityMatmul : Pass<"lower-pad-to-identity-matmul", "ModuleOp"> {
  let summary = "Lower tensor.pad to Left/Right Identity PCMM (linalg.matmul with cleartext identity weights).";
  let description = [{
    Rewrites static zero `tensor.pad` operations on secret tensors to explicit
    `linalg.matmul` (or `linalg.batch_matmul`) using cleartext identity diagonal
    matrices ($P_L \cdot X$ for row padding, $X \cdot P_R$ for column padding).

    This avoids irregular shift networks during layout conversion and leverages
    the structured BSGS rotate-and-reduce schedules ($2\sqrt{n}$ Galois keys)
    of HEIR's existing Left and Right PCMM kernels.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LOWERPADTOIDENTITYMATMUL_LOWERPADTOIDENTITYMATMUL_TD_
