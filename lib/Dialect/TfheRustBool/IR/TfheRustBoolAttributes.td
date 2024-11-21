#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_

include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"

class TfheRustBool_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TfheRustBool_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def TfheRustBoolGates_Attr : TfheRustBool_Attr<"TfheRustBoolGates", "tfhe_rust_bool_gates"> {
  let summary = "An Attribute containing an array of strings to store bool gates";

  let description = [{
     This attributes stores a list of integer identifiers for Boolean gates.

     Uses following mapping:
     AND_GATE = 0;
     NAND_GATE = 1;
     OR_GATE = 2;
     NOR_GATE = 3;
     XOR_GATE = 4;
     XNOR_GATE = 5;
     NOT_GATE = 6;
  }];

  let parameters = (ins
      ArrayRefParameter<"::mlir::heir::tfhe_rust_bool::TfheRustBoolGateEnumAttr">: $gates);

  let assemblyFormat = "`<` $gates `>`";
}




#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_
