#ifndef LIB_DIALECT_ARITH_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_ARITH_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def EmuWideInth : Pass<"emu-wide"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_ARITH_TRANSFORMS_PASSES_TD_
