#ifndef LIB_DIALECT_MATHEXT_IR_MATHEXTDIALECT_TD_
#define LIB_DIALECT_MATHEXT_IR_MATHEXTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def MathExt_Dialect : Dialect {
  let name = "math_ext";
  let description = [{
    // FIXME: add documentation
    The `math_ext` dialect is ...
  }];

  let cppNamespace = "::mlir::heir::math_ext";
}

#endif  // LIB_DIALECT_MATHEXT_IR_MATHEXTDIALECT_TD_
