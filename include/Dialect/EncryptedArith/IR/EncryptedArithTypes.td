#ifndef THIRD_PARTY_HEIR_INCLUDE_DIALECT_ENCRYPTEDARITH_IR_ENCRYPTEDARITHTYPES_TD_
#define THIRD_PARTY_HEIR_INCLUDE_DIALECT_ENCRYPTEDARITH_IR_ENCRYPTEDARITHTYPES_TD_

include "EncryptedArithDialect.td"

//===----------------------------------------------------------------------===//
// EncryptedArith type definitions
//===----------------------------------------------------------------------===//

// A base class for all types in this dialect
class EncryptedArith_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EncryptedArith_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EncryptedArith_EncInt : EncryptedArith_Type<"EncInt", "enc_int"> {
  let summary = "Encrypted integer type with arbitrary width";
  let description = [{
    An encrypted integer with variable width.
  }];

  let parameters = (ins "unsigned":$width);

  /// Example:
  ///
  ///    !enc_arith.enc_int<10> // a encrypted 10-bit integer.
  ///
  let assemblyFormat = "`<` $width `>`";
}

#endif  // THIRD_PARTY_HEIR_INCLUDE_DIALECT_ENCRYPTEDARITH_IR_ENCRYPTEDARITHTYPES_TD_
