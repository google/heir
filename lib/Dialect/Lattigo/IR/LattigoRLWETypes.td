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
  let nameSuggestion = "kgen";
}

def Lattigo_RLWESecretKey : Lattigo_RLWEType<"SecretKey", "secret_key"> {
  let description = [{
    This type represents the secret key for the RLWE encryption scheme.
  }];
  let nameSuggestion = "sk";
}

def Lattigo_RLWEPublicKey : Lattigo_RLWEType<"PublicKey", "public_key"> {
  let description = [{
    This type represents the public key for the RLWE encryption scheme.
  }];
  let nameSuggestion = "pk";
}

def Lattigo_RLWEPublicKeyOrSecretKey : AnyTypeOf<[Lattigo_RLWEPublicKey, Lattigo_RLWESecretKey]>;

def Lattigo_RLWERelinearizationKey : Lattigo_RLWEType<"RelinearizationKey", "relinearization_key"> {
  let description = [{
    This type represents the relinearization key for the RLWE encryption scheme.
  }];
  let nameSuggestion = "rk";
}

def Lattigo_RLWEGaloisKey : Lattigo_RLWEType<"GaloisKey", "galois_key"> {
  let description = [{
    This type represents the Galois key for the RLWE encryption scheme.

    galoisElement: Enabling the automorphism X -> X^{galoisElement}.
  }];
  let parameters = (ins Builtin_IntegerAttr:$galoisElement);
  let assemblyFormat = "`<` struct(params) `>`";
  let nameSuggestion = "gk";
}

def Lattigo_RLWEEvaluationKeySet : Lattigo_RLWEType<"EvaluationKeySet", "evaluation_key_set"> {
  let description = [{
    This type represents the evaluation key set for the RLWE encryption scheme.
  }];
  let nameSuggestion = "ekset";
}

def Lattigo_RLWEEncryptor : Lattigo_RLWEType<"Encryptor", "encryptor"> {
  let description = [{
    This type represents the encryptor for the RLWE encryption scheme.
  }];
  let parameters = (ins "bool":$publicKey);
  let assemblyFormat = "`<` struct(params) `>`";
  let nameSuggestion = "encryptor";
}

def Lattigo_RLWEDecryptor : Lattigo_RLWEType<"Decryptor", "decryptor"> {
  let description = [{
    This type represents the decryptor for the RLWE encryption scheme.
  }];
  let nameSuggestion = "decryptor";
}

def Lattigo_RLWEPlaintext : Lattigo_RLWEType<"Plaintext", "plaintext"> {
  let description = [{
    This type represents the plaintext for the RLWE encryption scheme.
  }];
  let nameSuggestion = "pt";
}

def Lattigo_RLWECiphertext : Lattigo_RLWEType<"Ciphertext", "ciphertext"> {
  let description = [{
    This type represents the ciphertext for the RLWE encryption scheme.
  }];
  let nameSuggestion = "ct";
}

def Lattigo_RLWECiphertextOrPlaintext : AnyTypeOf<[Lattigo_RLWECiphertext, Lattigo_RLWEPlaintext]>;

// common interface for RLWE
def Lattigo_RLWEParameter : AnyTypeOf<[Lattigo_BGVParameter, Lattigo_CKKSParameter]>;
def Lattigo_RLWEEncoder : AnyTypeOf<[Lattigo_BGVEncoder, Lattigo_CKKSEncoder]>;
def Lattigo_RLWEEvaluator : AnyTypeOf<[Lattigo_BGVEvaluator, Lattigo_CKKSEvaluator]>;

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGORLWETYPES_TD_
