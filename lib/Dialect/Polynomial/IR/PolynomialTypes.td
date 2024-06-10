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

  let parameters = (ins Ring_Attr:$ring);
  let assemblyFormat = "`<` $ring `>`";
}

def PolynomialLike: TypeOrContainer<Polynomial, "polynomial-like">;

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
