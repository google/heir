#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_

include "LattigoDialect.td"
include "LattigoAttributes.td"
include "lib/Utils/Tablegen/AsmInterfaces.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

// A base class for all types in this dialect
class Lattigo_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Lattigo_Dialect, name, traits # [
      DeclareTypeInterfaceMethods<TypeAsmInterface, ["suggestedName"]>
    ]> {
  let mnemonic = typeMnemonic;

  string nameSuggestion = ?;
  let extraClassDeclaration = "std::string suggestedName() {return \"" # nameSuggestion # "\"; }";
}

include "LattigoBGVTypes.td"
include "LattigoCKKSTypes.td"
include "LattigoRLWETypes.td"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_
