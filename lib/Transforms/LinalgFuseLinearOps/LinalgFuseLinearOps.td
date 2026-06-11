#ifndef LIB_TRANSFORMS_LINALGFUSELINEAROPS_LINALGFUSELINEAROPS_TD_
#define LIB_TRANSFORMS_LINALGFUSELINEAROPS_LINALGFUSELINEAROPS_TD_

include "mlir/Pass/PassBase.td"

def LinalgFuseLinearOps : Pass<"linalg-fuse-linear-ops", "ModuleOp"> {
  let summary = "Fuse arith ops into preceding linear operations.";
  let description = [{
    Fuses elementary arithmetic operations into preceding linalg operations when possible.

    In particular, this pass fuses:

    - arith.addf/subf into the `outs` of a preceding linalg op that supports
      DestinationStyleOpInterface.
    - arith.mulf/divf into the weights of an opt-in list of preceding linalg
      ops. For convolutions, only fusion by broadcasting along the channel
      dimension is supported.

    In the context of a neural network, this pass generalizes the task of
    folding a BatchNorm layer into a preceding dense or convolutional layer.

    (* example filepath=tests/Transforms/linalg_fuse_linear_ops/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // LIB_TRANSFORMS_LINALGFUSELINEAROPS_LINALGFUSELINEAROPS_TD_
