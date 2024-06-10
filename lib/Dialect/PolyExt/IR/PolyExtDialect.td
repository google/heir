#ifndef HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTDIALECT_TD_
#define HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// PolyExt dialect definitions
//===----------------------------------------------------------------------===//

def PolyExt_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "poly_ext";

  let description = [{
    The PolyExt dialect is a place for "external" operations on
    `mlir::polynomial::Polynomial`.
  }];

  let cppNamespace = "::mlir::heir::poly_ext";
}

#endif  // HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTDIALECT_TD_
