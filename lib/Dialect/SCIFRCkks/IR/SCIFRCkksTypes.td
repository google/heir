//===- SCIFRCkksTypes.td - SCIFRCkks Dialect Types -----------*- tablegen -*-===//

#ifndef SCIFRCkks_TYPES
#define SCIFRCkks_TYPES

include "SCIFRCkksDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class SCIFRCkks_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SCIFRCkks_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SCIFRCkks_Ciphertext : SCIFRCkks_Type<"SCIFRCkksCiphertext", "ciphertext"> {
  let summary = "A type for SCIFRCkks Ciphertext";
}

def SCIFRCkks_Bootstrap_Key_Standard : SCIFRCkks_Type<"SCIFRCkksBootstrapKeyStandard", "bootstrap_key_standard"> {
  let summary = "The key required to perform Bootstrap operation in SCIFRCkks.";
}

def SCIFRCkks_Key_Switch_Key : SCIFRCkks_Type<"SCIFRCkksKeySwitchKey", "key_switch_key"> {
  let summary = "The key required to perform Keyswitch operation in SCIFRCkks.";
}

def SCIFRCkks_Server_Parameters : SCIFRCkks_Type<"SCIFRCkksServerParameters", "server_parameters"> {
  let summary = "The server parameters required to map to cornami hardware in SCIFRCkks.";
}

#endif // SCIFRCkks_Types
