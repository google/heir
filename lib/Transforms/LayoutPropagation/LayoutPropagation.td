#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def LayoutPropagation : Pass<"layout-propagation"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
