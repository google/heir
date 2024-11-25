#ifndef LIB_DIALECT_RNS_IR_RNSTYPEINTERFACES_TD_
#define LIB_DIALECT_RNS_IR_RNSTYPEINTERFACES_TD_

include "mlir/IR/BuiltinTypeInterfaces.td"

def RNSBasisTypeInterface : TypeInterface<"RNSBasisTypeInterface"> {
  let cppNamespace = "::mlir::heir::rns";
  let description = [{
    This interface is required for a type to be used as a parameter
    to an `rns` type.
  }];
  let methods = [
    InterfaceMethod<
    /*description=*/[{
      Returns true if this type is compatible with another type in the
      same RNS basis. In particular, the set of types used for a single
      RNS basis are never equal as types, but instead have some common
      attribute that must be checked here. For example, an RNS type where
      the basis types are polynomials would return true if the two types
      are both polynomial types, even if they have different coefficient
      moduli.

      Another example is using mod arith types as the basis types, where
      by the nature of chinese reminder theorem, it is required that
      the modulus of them must be mutually coprime.

      `isCompatibleWith` must be commutative, in the sense
      that `type1.isCompatibleWith(type2)` if and only if
      `type2.isCompatibleWith(type1)`.
    }],
    "bool", "isCompatibleWith", (ins "::mlir::Type":$otherRnsBasisType)>
  ];
}

#endif  // LIB_DIALECT_RNS_IR_RNSTYPEINTERFACES_TD_
