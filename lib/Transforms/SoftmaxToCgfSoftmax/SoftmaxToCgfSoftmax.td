#ifndef LIB_TRANSFORMS_SOFTMAXTOCGFSOFTMAX_SOFTMAXTOCGFSOFTMAX_TD_
#define LIB_TRANSFORMS_SOFTMAXTOCGFSOFTMAX_SOFTMAXTOCGFSOFTMAX_TD_

include "mlir/Pass/PassBase.td"

def SoftmaxToCgfSoftmax : Pass<"softmax-to-cgf-softmax"> {
  let summary = "Lower math_ext.softmax to CGF-softmax approximation";
  let description = [{
    Lowers `math_ext.softmax` to a sequence of operations implementing
    the second-order CGF-softmax approximation. This eliminates
    division and max operations.

    For details on the approximation, see: https://arxiv.org/abs/2602.01621
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::math_ext::MathExtDialect"
  ];
}

#endif  // LIB_TRANSFORMS_SOFTMAXTOCGFSOFTMAX_SOFTMAXTOCGFSOFTMAX_TD_
