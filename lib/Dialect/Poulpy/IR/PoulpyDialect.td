#ifndef LIB_DIALECT_POULPY_IR_POULPYDIALECT_TD_
#define LIB_DIALECT_POULPY_IR_POULPYDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Poulpy_Dialect : Dialect {
  let name = "poulpy";
  let description = [{
  The `poulpy` dialect is an exit dialect for generating Rust code against the Poulpy library API.

  See https://github.com/poulpy-fhe/poulpy
  }];

  let cppNamespace = "::mlir::heir::poulpy";

  let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_POULPY_IR_POULPYDIALECT_TD_
