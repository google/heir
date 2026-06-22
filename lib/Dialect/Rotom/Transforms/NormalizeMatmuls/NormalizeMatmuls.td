#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_NORMALIZEMATMULS_NORMALIZEMATMULS_TD_
#define LIB_DIALECT_ROTOM_TRANSFORMS_NORMALIZEMATMULS_NORMALIZEMATMULS_TD_

include "mlir/Pass/PassBase.td"

def NormalizeMatmuls
    : Pass<"rotom-normalize-matmuls", "ModuleOp"> {
  let summary = "Normalize vector-times-matrix matmuls for the Rotom diagonal kernel";
  let description = [{
    Rewrites a row-vector-times-matrix matmul `x(1xK) * M(KxN)` (N > 1) into
    `(M^T * x^T)^T`, i.e. `transpose(matmul(M^T, x^T))`. The Rotom ciphertext-axis
    diagonal matvec kernel only handles matrix * column-vector (`M'xK * Kx1`); a
    neural-net layer `y = x . W^T` is exactly this transposed orientation, so the
    rewrite lets the existing kernel apply with the result transposed back. When
    `M` is itself a transpose (the common `y = x . W^T` case) its input is reused
    directly, avoiding a redundant transpose-of-transpose.

    Must run before rotom-seed-layout so the seeds land on the rewritten operands.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_NORMALIZEMATMULS_NORMALIZEMATMULS_TD_
