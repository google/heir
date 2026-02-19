#ifndef LIB_DIALECT_DEBUG_IR_DEBUGDIALECT_TD_
#define LIB_DIALECT_DEBUG_IR_DEBUGDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Debug_Dialect : Dialect {
  let name = "debug";

  let description = [{
    The Debug dialect contains operations for debugging and validation.

    These ops are intended to be used during development to do things like
    insert function calls at specific program checkpoints.
  }];

  let cppNamespace = "::mlir::heir::debug";
}

#endif  // LIB_DIALECT_DEBUG_IR_DEBUGDIALECT_TD_
