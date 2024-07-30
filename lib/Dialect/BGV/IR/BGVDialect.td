#ifndef LIB_DIALECT_BGV_IR_BGVDIALECT_TD_
#define LIB_DIALECT_BGV_IR_BGVDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def BGV_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "bgv";

  let description = [{
    The BGV dialect defines the types and operations of the BGV cryptosystem.
  }];

  let cppNamespace = "::mlir::heir::bgv";
}

#endif  // LIB_DIALECT_BGV_IR_BGVDIALECT_TD_
