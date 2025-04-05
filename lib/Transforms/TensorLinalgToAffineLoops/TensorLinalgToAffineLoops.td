#ifndef LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_TD_
#define LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_TD_

include "mlir/Pass/PassBase.td"

def TensorLinalgToAffineLoops : Pass<"tensor-linalg-to-affine-loops"> {
  let summary = "A port of convert-linalg-to-affine-loops for loops with tensor semantics";
  let description = [{
  This pass primarily exists to support the conversion of `linalg.generic`
  operations that implement `tensor_ext.assign_layout` ops.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_TD_
