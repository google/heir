#ifndef LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_TD_
#define LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_TD_

include "mlir/Pass/PassBase.td"

def LowerUnpack : Pass<"lower-unpack"> {
  let summary = "Lower tensor_ext.unpack to standard MLIR";
  let description = [{
  This pass lowers tensor_ext.unpack.
  }];
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_TD_
