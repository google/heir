#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_

include "TfheRustDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class TfheRust_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TfheRust_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PassByReference : NativeTypeTrait<"PassByReference"> {
  let cppNamespace = "::mlir::heir::tfhe_rust";
}

def EncryptedInteger : NativeTypeTrait<"EncryptedInteger"> {
  let cppNamespace = "::mlir::heir::tfhe_rust";
}

class TfheRust_EncryptedUInt<int width>
  : TfheRust_Type<"EncryptedUInt" # width, "eui" # width, [PassByReference, EncryptedInteger, MemRefElementTypeInterface]> {
  let summary = "An encrypted unsigned integer corresponding to tfhe-rs's FHEUint" # width # " type";
}

// Available options are https://docs.rs/tfhe/latest/tfhe/index.html#types
foreach i = [2, 3, 4, 8, 10, 12, 14, 16, 32, 64, 128, 256] in {
  def TfheRust_EncryptedUInt # i : TfheRust_EncryptedUInt<i>;
}

class TfheRust_EncryptedInt<int width>
  : TfheRust_Type<"EncryptedInt" # width, "ei" # width, [PassByReference, EncryptedInteger]> {
  let summary = "An encrypted signed integer corresponding to tfhe-rs's FHEInt" # width # " type";
}

// Available options are https://docs.rs/tfhe/latest/tfhe/index.html#types
foreach i = [8, 16, 32, 64, 128, 256] in {
  def TfheRust_EncryptedInt # i : TfheRust_EncryptedInt<i>;
}

def TfheRust_CiphertextType :
  AnyTypeOf<[
    TfheRust_EncryptedUInt2,
    TfheRust_EncryptedUInt3,
    TfheRust_EncryptedUInt4,
    TfheRust_EncryptedUInt8,
    TfheRust_EncryptedUInt10,
    TfheRust_EncryptedUInt12,
    TfheRust_EncryptedUInt14,
    TfheRust_EncryptedUInt16,
    TfheRust_EncryptedUInt32,
    TfheRust_EncryptedUInt64,
    TfheRust_EncryptedUInt128,
    TfheRust_EncryptedUInt256,
    TfheRust_EncryptedInt8,
    TfheRust_EncryptedInt16,
    TfheRust_EncryptedInt32,
    TfheRust_EncryptedInt64,
    TfheRust_EncryptedInt128,
    TfheRust_EncryptedInt256,
  ]>;


def TfheRust_ServerKey : TfheRust_Type<"ServerKey", "server_key", [PassByReference]> {
  let summary = "The short int server key required to perform homomorphic operations.";
}

def TfheRust_LookupTable : TfheRust_Type<"LookupTable", "lookup_table", [PassByReference]> {
  let summary = "A univariate lookup table used for programmable bootstrapping.";
}

#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
