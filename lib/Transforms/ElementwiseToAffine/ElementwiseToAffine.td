#ifndef LIB_TRANSFORMS_ELEMENTWISETOAFFINE_ELEMENTWISETOAFFINE_TD_
#define LIB_TRANSFORMS_ELEMENTWISETOAFFINE_ELEMENTWISETOAFFINE_TD_

include "mlir/Pass/PassBase.td"

def ElementwiseToAffine : Pass<"convert-elementwise-to-affine"> {
  let summary = "This pass lowers ElementwiseMappable operations to Affine loops.";
  let description = [{
    This pass lowers ElementwiseMappable operations over tensors
    to affine loop nests that instead apply the operation to the underlying scalar values.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // LIB_TRANSFORMS_ELEMENTWISETOAFFINE_ELEMENTWISETOAFFINE_TD_
