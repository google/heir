#ifndef LIB_DIALECT_ROTOM_IR_ROTOMDIALECT_TD_
#define LIB_DIALECT_ROTOM_IR_ROTOMDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Rotom_Dialect : Dialect {
  let name = "rotom";
  let description = [{
    Dialect for Rotom's layout representation.
    See the Rotom paper: https://eprint.iacr.org/2025/1319
  }];
  let cppNamespace = "::mlir::heir::rotom";

  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMDIALECT_TD_
