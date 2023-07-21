#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_

include "PolyDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Poly type definitions
//===----------------------------------------------------------------------===//

// A base class for all types in this dialect
class Poly_Type<string name, string typeMnemonic>
    : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Poly : Poly_Type<"Polynomial", "poly"> {
  let summary = "Polynomial represents an element of a polynomial quotient ring";

  let description = [{
    A type for polynomials in a polynomial quotient ring.

    The type is parametrized by the degree of the polynomial

  }];
}

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_
