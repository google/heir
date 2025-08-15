#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_TD_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_TD_

include "mlir/Pass/PassBase.td"

def LayoutOptimization : Pass<"layout-optimization"> {
  let summary = "Optimize layout conversions in the IR";
  let description = [{
  This pass performance a greedy layout optimization pass similar to the
  automatic layout assignment from [A Tensor Compiler with Automatic Data
  Packing for Simple and Efficient Fully Homomorphic
  Encryption](https://dl.acm.org/doi/pdf/10.1145/3656382). The pass assumes that
  an initial layout assignment was provided on each operation through the
  `layout-propagation` pass.

  The pass iterates on each operation of the IR in reverse order, attempting to
  hoist a layout conversion of the operation's result before the operation. For
  each of the result's layout conversions, the pass will compute the net cost of
  hoisting the conversion through the operation by considering the following:
    1. The cost of performing the operation with new input layouts that result
       in the desired layout.
    2. The cost of the converting the layout of each input.
    3. The new cost of converting from the desired layout to each other layout
       conversions of the result.

  The layout conversion that results in the lowest net cost is chosen to be
  hoisted.

  (* example filepath=tests/Transforms/layout_optimization/hoist_matvec.mlir *)
  ```
  }];

  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
  ];

  let options = [
    Option<
      "ciphertextSize",
      "ciphertext-size",
      "int",
      /*default=*/"1024",
      "Power of two length of the ciphertexts the data is packed in."
    >
  ];
}

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_TD_
