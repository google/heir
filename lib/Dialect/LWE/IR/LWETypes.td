#ifndef LIB_DIALECT_LWE_IR_LWETYPES_TD_
#define LIB_DIALECT_LWE_IR_LWETYPES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/LWEAttributes.td"
include "lib/Dialect/LWE/IR/NewLWETypes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

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
  let asmName = "ct";
}

def LWECiphertextLike : TypeOrValueSemanticsContainer<NewLWECiphertext, "ciphertext-like">;

def RLWECiphertext : LWE_Type<"RLWECiphertext", "rlwe_ciphertext"> {
  let summary = "A type for RLWE ciphertexts";
  let parameters = (ins
    "::mlir::Attribute":$encoding,
    "RLWEParamsAttr":$rlwe_params,
    "Type":$underlying_type
  );
  let asmName = "ct";
}

def RLWECiphertextLike : TypeOrValueSemanticsContainer<RLWECiphertext, "ciphertext-like">;

def RLWESecretKey : LWE_Type<"RLWESecretKey", "rlwe_secret_key"> {
  let summary = "A secret key for RLWE";
  let parameters = (ins "RLWEParamsAttr":$rlwe_params);
  let asmName = "sk";
}

def RLWEPublicKey : LWE_Type<"RLWEPublicKey", "rlwe_public_key"> {
  let summary = "A public key for RLWE";
  let parameters = (ins "RLWEParamsAttr":$rlwe_params);
  let asmName = "pk";
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
  let asmName = "pt";
}

def LWEPlaintextLike : TypeOrValueSemanticsContainer<LWEPlaintext, "lwe-plaintext-like">;

def RLWEPlaintext : LWE_Type<"RLWEPlaintext", "rlwe_plaintext"> {
  let summary = "A type for RLWE plaintexts";

  let parameters = (ins
    "::mlir::Attribute":$encoding,
    "::mlir::heir::polynomial::RingAttr":$ring,
    "Type":$underlying_type
  );
  let asmName = "pt";
}

def RLWEPlaintextLike : TypeOrValueSemanticsContainer<RLWEPlaintext, "rlwe-plaintext-like">;


#endif  // LIB_DIALECT_LWE_IR_LWETYPES_TD_
