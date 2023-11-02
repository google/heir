#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_

include "TfheRustDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

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


#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
