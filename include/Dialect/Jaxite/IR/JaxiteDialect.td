#ifndef INCLUDE_DIALECT_JAXITE_IR_JAXITEDIALECT_H_
#define INCLUDE_DIALECT_JAXITE_IR_JAXITEDIALECT_H_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Jaxite_Dialect : Dialect {
  let name = "jaxite";

  let description = [{
    The `jaxite` dialect is an exit dialect for generating py code against the jaxite library API,
    using the jaxite parameters and encoding scheme.

    See https://github.com/google/jaxite
  }];

  let cppNamespace = "::mlir::heir::jaxite";

  let useDefaultTypePrinterParser = 1;
}

#endif  // INCLUDE_DIALECT_JAXITE_IR_JAXITEDIALECT_H_
