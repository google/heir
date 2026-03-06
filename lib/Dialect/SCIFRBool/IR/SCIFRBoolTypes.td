//===- SCIFRBoolTypes.td - SCIFRBool Dialect Types -----------*- tablegen -*-===//

#ifndef SCIFRBool_TYPES
#define SCIFRBool_TYPES

include "SCIFRBoolDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class SCIFRBool_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SCIFRBool_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SCIFRBool_Bootstrap_Key : SCIFRBool_Type<"SCIFRBoolBootstrapKey", "bootstrap_key"> {
  let summary = "The key required to perform Bootstrap operation in SCIFRBool.";
}

def SCIFRBool_Key_Switch_Key : SCIFRBool_Type<"SCIFRBoolKeySwitchKey", "key_switch_key"> {
  let summary = "The key required to perform Keyswitch operation in SCIFRBool.";
}

def SCIFRBool_Server_Parameters : SCIFRBool_Type<"SCIFRBoolServerParameters", "server_parameters"> {
  let summary = "The server parameters required to map to cornami hardware in SCIFRBool.";
}

#endif // SCIFRBool_Types
