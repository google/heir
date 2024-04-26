#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEDIALECT_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Openfhe_Dialect : Dialect {
  let name = "openfhe";

  let description = [{
    The `openfhe` dialect is an exit dialect for generating c++ code against the OpenFHE library API.

    See https://github.com/openfheorg/openfhe-development
  }];

  let cppNamespace = "::mlir::heir::openfhe";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEDIALECT_TD_
