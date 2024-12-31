#ifndef LIB_DIALECT_PISA_IR_PISADIALECT_TD_
#define LIB_DIALECT_PISA_IR_PISADIALECT_TD_

include "mlir/IR/DialectBase.td"

def PISA_Dialect : Dialect {
  let name = "pisa";
  let description = [{
    // FIXME: add documentation
    The `pisa` dialect is ...
  }];

  let cppNamespace = "::mlir::heir::pisa";
}

#endif  // LIB_DIALECT_PISA_IR_PISADIALECT_TD_
