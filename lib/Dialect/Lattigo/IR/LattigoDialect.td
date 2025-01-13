#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGODIALECT_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGODIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Lattigo_Dialect : Dialect {
  let name = "lattigo";
  let description = [{
    The `lattigo` dialect is an exit dialect for generating GO code against the Lattigo library API.

    See https://github.com/tuneinsight/lattigo
  }];

  let cppNamespace = "::mlir::heir::lattigo";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGODIALECT_TD_
