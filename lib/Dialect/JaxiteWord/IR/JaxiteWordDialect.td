#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDDIALECT_H_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDDIALECT_H_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def JaxiteWord_Dialect : Dialect {
  let name = "jaxiteword";

  let description = [{
    The `jaxiteword` dialect is an exit dialect for generating py code against the jaxiteword library API,
    using the jaxiteword parameters and encoding scheme.

    See https://github.com/google/jaxite/jaxite_word
  }];

  let cppNamespace = "::mlir::heir::jaxiteword";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDDIALECT_H_
