#ifndef INCLUDE_DIALECT_RNS_IR_RNSTYPES_TD_
#define INCLUDE_DIALECT_RNS_IR_RNSTYPES_TD_

include "include/Dialect/RNS/IR/RNSDialect.td"
include "include/Dialect/RNS/IR/RNSTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class RNS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RNS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def RNS : RNS_Type<"RNS", "rns"> {
  let summary = "A residue number system representation";
  let description = [{
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$basisTypes);
  let assemblyFormat = "`<` $basisTypes `>`";
  let genVerifyDecl = 1;
}

#endif  // INCLUDE_DIALECT_RNS_IR_RNSTYPES_TD_
