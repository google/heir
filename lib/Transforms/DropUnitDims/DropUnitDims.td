#ifndef LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_TD_
#define LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_TD_

include "mlir/Pass/PassBase.td"

def DropUnitDims : Pass<"drop-unit-dims"> {
  let summary = "Drops unit dimensions from linalg ops.";
  let description = [{
  This pass converts `linalg` whose operands have unit dimensions
  in their types to specialized ops that drop these unit dimensions.

  For example, a `linalg.matmul` whose RHS has type `tensor<32x1xi32>` is
  converted to a `linalg.matvec` op on the underlying `tensor<32xi32>`.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

#endif  // LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_TD_
