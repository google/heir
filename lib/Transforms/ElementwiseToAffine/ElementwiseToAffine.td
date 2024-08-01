#ifndef LIB_TRANSFORMS_ELEMENTWISETOAFFINE_ELEMENTWISETOAFFINE_TD_
#define LIB_TRANSFORMS_ELEMENTWISETOAFFINE_ELEMENTWISETOAFFINE_TD_

include "mlir/Pass/PassBase.td"

def ElementwiseToAffine : Pass<"convert-elementwise-to-affine"> {
  let summary = "This pass lowers ElementwiseMappable operations to Affine loops.";
  let description = [{
    This pass lowers ElementwiseMappable operations over tensors
    to affine loop nests that instead apply the operation to the underlying scalar values.

    Usage:
    '--convert-elementwise-to-affine=convert-ops=arith.mulf '
      restrict conversion to mulf op from arith dialect.

    '--convert-elementwise-to-affine=convert-ops=arith.addf,arith.divf  convert-dialects=bgv'
      restrict conversion to addf and divf ops from arith dialect and all of the ops in bgv dialect.

    --convert-elementwise-to-affine=convert-dialects=arith
      restrict conversion to arith dialect so ops only from arith dialect is processed.

    --convert-elementwise-to-affine=convert-ops=arith.addf,arith.mulf
      restrict conversion only to these two ops - addf and mulf - from arith dialect.

  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::tensor::TensorDialect"
  ];
  let options = [
    ListOption<"convertOps","convert-ops", "std::string",
                "comma-separated list of ops to run this pass on ">,
    ListOption<"convertDialects","convert-dialects", "std::string",
                "comma-separated list of dialects to run this pass on ">,
  ];
}

#endif  // LIB_TRANSFORMS_ELEMENTWISETOAFFINE_ELEMENTWISETOAFFINE_TD_
