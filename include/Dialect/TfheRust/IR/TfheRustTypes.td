#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_

include "TfheRustDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// A base class for all types in this dialect
class TfheRust_Type<string name, string typeMnemonic>
    : TypeDef<TfheRust_Dialect, name> {
  let mnemonic = typeMnemonic;
}


class TfheRust_EncryptedUInt<int width> : TfheRust_Type<"TfheRust_EncryptedUInt" # width, "eui" # width> {
  let summary = "An encrypted unsigned integer corresponding to tfhe-rs's FHEUint" # width # " type.";
}

// Available options are https://docs.rs/tfhe/latest/tfhe/index.html#types
foreach i = [2, 3, 4, 8, 10, 12, 14, 16, 32, 64, 128, 256] in {
  def TfheRust_EncryptedUInt # i : TfheRust_EncryptedUInt<i>;
}

class TfheRust_EncryptedInt<int width> : TfheRust_Type<"TfheRust_EncryptedInt" # width, "ei" # width> {
  let summary = "An encrypted signed integer corresponding to tfhe-rs's FHEInt" # width # " type.";
}

// Available options are https://docs.rs/tfhe/latest/tfhe/index.html#types
foreach i = [8, 16, 32, 64, 128, 256] in {
  def TfheRust_EncryptedInt # i : TfheRust_EncryptedInt<i>;
}

def ServerKey : TfheRust_Type<"ServerKey", "server_key"> {
  let summary = "The server key required to perform homomorphic operations.";
}

def TfheRust_CiphertextType :
  AnyTypeOf<[
    EncryptedUInt2,
    EncryptedUInt3,
    EncryptedUInt4,
    EncryptedUInt8,
    EncryptedUInt10,
    EncryptedUInt12,
    EncryptedUInt14,
    EncryptedUInt16,
    EncryptedUInt32,
    EncryptedUInt64,
    EncryptedUInt128,
    EncryptedUInt256,
    EncryptedInt8,
    EncryptedInt16,
    EncryptedInt32,
    EncryptedInt64,
    EncryptedInt128,
    EncryptedInt256,
  ]>;

#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
