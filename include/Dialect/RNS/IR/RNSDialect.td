#ifndef INCLUDE_DIALECT_RNS_IR_RNSDIALECT_TD_
#define INCLUDE_DIALECT_RNS_IR_RNSDIALECT_TD_

include "mlir/IR/DialectBase.td"

def RNS_Dialect : Dialect {
  let name = "rns";
  let description = [{
    // FIXME: add documentation
    The `rns` dialect is ...
  }];

  let cppNamespace = "::mlir::heir::rns";
}

#endif  // INCLUDE_DIALECT_RNS_IR_RNSDIALECT_TD_
