#ifndef INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
#define INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_

include "include/Dialect/Polynomial/IR/PolynomialDialect.td"
include "include/Dialect/Polynomial/IR/PolynomialAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Polynomial_Type<string name, string typeMnemonic>
    : TypeDef<Polynomial_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Polynomial_Type<"Polynomial", "polynomial"> {
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

#endif  // INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
