#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTONEWMOD_TD_
#define LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTONEWMOD_TD_

include "mlir/Pass/PassBase.td"

def ArithToNewMod : Pass<"arith-to-new", "mlir::ModuleOp"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
    let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTONEWMOD_TD_
