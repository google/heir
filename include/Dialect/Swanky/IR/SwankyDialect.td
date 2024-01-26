#ifndef HEIR_INCLUDE_DIALECT_Swanky_IR_SwankyDIALECT_TD_
#define HEIR_INCLUDE_DIALECT_Swanky_IR_SwankyDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Swanky_Dialect : Dialect {
  let name = "swanky";
  let description = [{
    // FIXME: add documentation
    The `swanky` dialect is ...
  }];

  let cppNamespace = "::mlir::heir::swanky";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_Swanky_IR_SwankyDIALECT_TD_
