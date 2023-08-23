#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEDIALECT_TD_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def LWE_Dialect : Dialect {
  let name = "lwe";
  let summary = "A dialect for the Learning With Errors (LWE) cryptosystem";
  let description = [{
    The `lwe` dialect defines the types and operations of the Learning With
    Errors (LWE) cryptosystem.

    See [Wikipedia](https://en.wikipedia.org/wiki/Learning_with_errors) for an
    overview of LWE.
  }];

  let cppNamespace = "::mlir::heir::lwe";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEDIALECT_TD_
