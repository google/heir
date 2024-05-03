#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_

include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/RNS/IR/RNSTypeInterfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

// A base class for all types in this dialect
class Polynomial_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Polynomial_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Polynomial_Type<"Polynomial", "polynomial",
    [DeclareTypeInterfaceMethods<RNSBasisTypeInterface>]> {
  let summary = "An element of a polynomial quotient ring";

  let description = [{
    A type for polynomials in a polynomial quotient ring.
  }];

  let parameters = (ins
    Ring_Attr:$ring,
    OptionalParameter<"Attribute">:$encoding
  );

  // TODO(#181): Skip the default builder since the custom builder enables the
  // default value for the encoding parameter, And I couldn't get
  // DefaultValueParameter to auto-generate a builder with the appropriate default
  // value.
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

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
