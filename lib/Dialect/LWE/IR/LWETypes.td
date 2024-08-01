#ifndef LIB_DIALECT_LWE_IR_LWETYPES_TD_
#define LIB_DIALECT_LWE_IR_LWETYPES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/LWEAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class LWE_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LWE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

// LWE Ciphertexts are ranked tensors of integers representing the LWE samples
// and the bias.
def LWECiphertext : LWE_Type<"LWECiphertext", "lwe_ciphertext", [MemRefElementTypeInterface]> {
  let summary = "A type for LWE ciphertexts";

  let description = [{
    A type for LWE ciphertexts.

    This type keeps track of the plaintext integer encoding for the LWE
    Ciphertext to ensure proper decoding after decryption. It also keeps track
    of the ring where the LWE ciphertext is defined, which provides information
    on the ciphertext shape and the ring operations used in LWE operations.
  }];

  let parameters = (ins
    "::mlir::Attribute":$encoding,
    OptionalParameter<"LWEParamsAttr">:$lwe_params
  );
}

def LWECiphertextLike : TypeOrContainer<LWECiphertext, "ciphertext-like">;

def RLWECiphertext : LWE_Type<"RLWECiphertext", "rlwe_ciphertext"> {
  let summary = "A type for RLWE ciphertexts";
  let parameters = (ins
    "::mlir::Attribute":$encoding,
    "RLWEParamsAttr":$rlwe_params,
    "Type":$underlying_type
  );
}

def RLWECiphertextLike : TypeOrContainer<RLWECiphertext, "ciphertext-like">;

def RLWESecretKey : LWE_Type<"RLWESecretKey", "rlwe_secret_key"> {
  let summary = "A secret key for RLWE";
  let parameters = (ins "RLWEParamsAttr":$rlwe_params);
}

def RLWEPublicKey : LWE_Type<"RLWEPublicKey", "rlwe_public_key"> {
  let summary = "A public key for RLWE";
  let parameters = (ins "RLWEParamsAttr":$rlwe_params);
}

def RLWESecretOrPublicKey : AnyTypeOf<[RLWESecretKey, RLWEPublicKey]>;

def LWEPlaintext : LWE_Type<"LWEPlaintext", "lwe_plaintext"> {
  let summary = "A type for LWE plaintexts";

  let description = [{
    A type for LWE plaintexts.

    This type keeps track of the plaintext integer encoding for the LWE
    plaintext before it is encrypted.
  }];

  let parameters = (ins
    "::mlir::Attribute":$encoding
  );
}

def RLWEPlaintext : LWE_Type<"RLWEPlaintext", "rlwe_plaintext"> {
  let summary = "A type for RLWE plaintexts";

  let parameters = (ins
    "::mlir::Attribute":$encoding,
    "::mlir::polynomial::RingAttr":$ring,
    "Type":$underlying_type
  );
}

#endif  // LIB_DIALECT_LWE_IR_LWETYPES_TD_
