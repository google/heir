#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_

include "include/Dialect/Poly/IR/PolyDialect.td"
include "include/Dialect/Poly/IR/PolyAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Poly_Type<string name, string typeMnemonic>
    : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Poly", "poly"> {
  let summary = "An element of a polynomial quotient ring";

  let description = [{
    A type for polynomials in a polynomial quotient ring.
  }];

  let parameters = (ins
    Ring_Attr:$ring,
    OptionalParameter<"Attribute">:$encoding
  );

  // TODO(https://github.com/google/heir/issues/181): Skip the default
  // builder since the custom builder enables the default value for the
  // encoding parameter, And I couldn't get DefaultValueParameter to
  // auto-generate a builder with the appropriate default value.
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins
      "RingAttr":$ring, CArg<"Attribute", "{}">:$encoding
    ), [{
      return $_get($_ctxt, ring, encoding);
    }]>
  ];

  let assemblyFormat = "`<` $ring (`,` $encoding^ )? `>`";
}

def PolynomialLike: TypeOrContainer<Polynomial, "polynomial-like">;

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYTYPES_TD_
