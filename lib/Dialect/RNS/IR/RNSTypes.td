#ifndef LIB_DIALECT_RNS_IR_RNSTYPES_TD_
#define LIB_DIALECT_RNS_IR_RNSTYPES_TD_

include "lib/Dialect/RNS/IR/RNSDialect.td"
include "lib/Dialect/RNS/IR/RNSTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"

// A base class for all types in this dialect
class RNS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RNS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def RNS : RNS_Type<"RNS", "rns", [OpAsmTypeInterface]> {
  let summary = "A residue number system representation";
  let description = [{
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$basisTypes);
  let assemblyFormat = "`<` $basisTypes `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Helper method for other Type/Attribute containing this Type.
    void getAliasSuffix(::llvm::raw_ostream& os) const {
      auto level = getBasisTypes().size() - 1; // start from 0
      os << "_L" << level;
    }

    // OpAsmTypeInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "rns";
      getAliasSuffix(os);
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def RNSLike: TypeOrValueSemanticsContainer<RNS, "rns-like">;

#endif  // LIB_DIALECT_RNS_IR_RNSTYPES_TD_
