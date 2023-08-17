#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVDIALECT_TD_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVDIALECT_TD_

include "mlir/IR/DialectBase.td"

def BGV_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "bgv";

  let summary = "A dialect for types and operations in the BGV cryptosystem";
  let description = [{
    The BGV dialect defines the types and operations of the BGV cryptosystem
  }];

  let cppNamespace = "::mlir::heir::bgv";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVDIALECT_TD_
