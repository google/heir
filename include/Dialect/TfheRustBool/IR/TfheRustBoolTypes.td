#ifndef INCLUDE_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLTYPES_TD_
#define INCLUDE_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLTYPES_TD_

include "TfheRustBoolDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class TfheRustBool_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TfheRustBool_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PassByReference : NativeTypeTrait<"PassByReference"> {
  let cppNamespace = "::mlir::heir::tfhe_rust_bool";
}

def TfheRustBool_Encrypted: TfheRustBool_Type<"EncryptedBool", "eb", [PassByReference]> {
  let summary = "An encrypted Boolean corresponding to tfhe-rs's FHEBool type";
}

def TfheRustBool_ServerKey : TfheRustBool_Type<"ServerKey", "server_key", [PassByReference]> {
  let summary = "The boolean server key required to perform homomorphic operations.";
}

#endif  // INCLUDE_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLTYPES_TD_
