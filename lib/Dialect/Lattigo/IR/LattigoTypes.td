#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_

include "LattigoDialect.td"
include "LattigoAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"

// A base class for all types in this dialect
class Lattigo_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Lattigo_Dialect, name, traits # [OpAsmTypeInterface]> {
  let mnemonic = typeMnemonic;

  string asmName = ?;
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("}] # asmName # [{");
    }
  }];
}

include "LattigoBGVTypes.td"
include "LattigoCKKSTypes.td"
include "LattigoRLWETypes.td"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_
