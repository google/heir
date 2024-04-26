#ifndef HEIR_LIB_DIALECT_CGGI_IR_CGGIDIALECT_TD_
#define HEIR_LIB_DIALECT_CGGI_IR_CGGIDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def CGGI_Dialect : Dialect {
  let name = "cggi";
  let summary = "A dialect for types and operations in the CGGI cryptosystem";
  let cppNamespace = "::mlir::heir::cggi";
  let useDefaultAttributePrinterParser = 1;
}

#endif  // HEIR_LIB_DIALECT_CGGI_IR_CGGIDIALECT_TD_
