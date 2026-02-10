#ifndef LIB_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def ModArith_Dialect : Dialect {
  let name = "mod_arith";
  let description = [{
    The `mod_arith` dialect contains operations used for modulo arithmetic.
  }];

  let cppNamespace = "::mlir::heir::mod_arith";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let dependentDialects = [
    "arith::ArithDialect",
    "mlir::heir::rns::RNSDialect"
  ];
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
