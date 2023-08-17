#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_

include "PolyDialect.td"
include "PolyAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Poly_Type<string name, string typeMnemonic>
    : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let summary = "An element of a polynomial quotient ring";

  let description = [{
    A type for polynomials in a polynomial quotient ring.
  }];

  let parameters = (ins Ring_Attr:$ring);
  let assemblyFormat = "`<` $ring `>`";
}

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_
