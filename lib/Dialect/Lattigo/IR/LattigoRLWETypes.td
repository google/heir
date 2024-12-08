#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGORLWETYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGORLWETYPES_TD_

include "LattigoAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

class Lattigo_RLWEType<string name, string typeMnemonic>
    : Lattigo_Type<"RLWE" # name, "rlwe." # typeMnemonic> {
}

def Lattigo_RLWEKeyGenerator : Lattigo_RLWEType<"KeyGenerator", "key_generator"> {
  let description = [{
    This type represents the key generator for the RLWE encryption scheme.
  }];
}

def Lattigo_RLWESecretKey : Lattigo_RLWEType<"SecretKey", "secret_key"> {
  let description = [{
    This type represents the secret key for the RLWE encryption scheme.
  }];
}

def Lattigo_RLWEPublicKey : Lattigo_RLWEType<"PublicKey", "public_key"> {
  let description = [{
    This type represents the public key for the RLWE encryption scheme.
  }];
}

def Lattigo_RLWEEncryptor : Lattigo_RLWEType<"Encryptor", "encryptor"> {
  let description = [{
    This type represents the encryptor for the RLWE encryption scheme.
  }];
}

def Lattigo_RLWEDecryptor : Lattigo_RLWEType<"Decryptor", "decryptor"> {
  let description = [{
    This type represents the decryptor for the RLWE encryption scheme.
  }];
}

def Lattigo_RLWEPlaintext : Lattigo_RLWEType<"Plaintext", "plaintext"> {
  let description = [{
    This type represents the plaintext for the RLWE encryption scheme.
  }];
}

def Lattigo_RLWECiphertext : Lattigo_RLWEType<"Ciphertext", "ciphertext"> {
  let description = [{
    This type represents the ciphertext for the RLWE encryption scheme.
  }];
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGORLWETYPES_TD_
