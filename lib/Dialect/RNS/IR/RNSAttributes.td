#ifndef LIB_DIALECT_RNS_IR_RNSATTRIBUTES_TD_
#define LIB_DIALECT_RNS_IR_RNSATTRIBUTES_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/RNS/IR/RNSDialect.td"
include "lib/Dialect/RNS/IR/RNSTypes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

class RNS_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RNS_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def RNS_RNSAttr : RNS_Attr<"RNS", "value", [DeclareAttrInterfaceMethods<LimbwiseAttrInterface>]> {
  let summary = "a typed RNS value";
  let description = [{
    A typed RNS value with one integer per basis limb.

    Example:

    ```mlir
    #v1 = #mod_arith.value<17 : !mod_arith.int<256 : i32>>
    #v2 = #mod_arith.value<19 : !mod_arith.int<256 : i32>>
    #v = #rns.value<[#v1, #v2]>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"Attribute", "One residue for each limb of the RNS basis.">:$values,
    AttrParameter<"::mlir::heir::rns::RNSType", "The RNS coefficient type.">:$type
  );
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static RNSAttr get(::mlir::heir::rns::RNSType type, ::llvm::ArrayRef<::mlir::Attribute> values) {
      return get(type.getContext(), values, type);
    }

  }];
}

#endif  // LIB_DIALECT_RNS_IR_RNSATTRIBUTES_TD_
