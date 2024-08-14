#ifndef LIB_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_

include "mlir/IR/DialectBase.td"

def ModArith_Dialect : Dialect {
  let name = "mod_arith";
  let description = [{
    The `mod_arith` dialect contains operations used for modulo arithmetic.
  }];

  let cppNamespace = "::mlir::heir::mod_arith";
  let dependentDialects = [
    "arith::ArithDialect",
  ];
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
