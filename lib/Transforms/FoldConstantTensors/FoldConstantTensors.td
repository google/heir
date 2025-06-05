#ifndef LIB_TRANSFORMS_FOLDCONSTANTTENSORS_FOLDCONSTANTTENSORS_TD_
#define LIB_TRANSFORMS_FOLDCONSTANTTENSORS_FOLDCONSTANTTENSORS_TD_

include "mlir/Pass/PassBase.td"

def FoldConstantTensors : Pass<"fold-constant-tensors"> {
  let summary = "This pass folds any constant tensors.";

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let description = [{
    This pass folds tensor operations on constants to new constants.

    The following folders are supported:
    * `tensor.insert` of a constant tensor
    * `tensor.collapse_shape` of a constant tensor
  }];
}

#endif  // LIB_TRANSFORMS_FOLDCONSTANTTENSORS_FOLDCONSTANTTENSORS_TD_
