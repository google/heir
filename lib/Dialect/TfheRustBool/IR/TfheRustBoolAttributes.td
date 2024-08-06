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

def TfheRustBoolGate_Attr : TfheRustBool_Attr<"TfheRustBoolGate", "tfhe_rust_bool_gate"> {
  let summary = "An Attribute containing an array of strings to store bool gates";

  let description = [{
    This attributes stores a list of string identifiers for Boolean gates.

     This used in the `tfhe_rust_bool.packed_gates` operation to indicate the boolean gate that applies pairwise between elements of two ciphertext arrays. For example,

     %0 = tfhe_rust_bool.packed_gates %a, %b {gates = #tfhe_rust_bool.tfhe_rust_bool_gate<"and", "xor">} : (!tfhe_rust_bool.server_key, tensor<2x!tfhe_rust_bool.eb>, tensor<2x!tfhe_rust_bool.eb>) -> tensor<2x!tfhe_rust_bool.eb>

     applies an "and" gate between the first elements of %a and %b and an xor gate between the second elements.
  }];

  let parameters = (ins
      ArrayRefParameter<"mlir::StringAttr">: $gate);

  let assemblyFormat = "`<` $gate `>`";
}




#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_
