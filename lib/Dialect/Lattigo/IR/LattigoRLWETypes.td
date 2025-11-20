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
  let asmName = "kgen";
}

def Lattigo_RLWESecretKey : Lattigo_RLWEType<"SecretKey", "secret_key"> {
  let description = [{
    This type represents the secret key for the RLWE encryption scheme.
  }];
  let asmName = "sk";
}

def Lattigo_RLWEPublicKey : Lattigo_RLWEType<"PublicKey", "public_key"> {
  let description = [{
    This type represents the public key for the RLWE encryption scheme.
  }];
  let asmName = "pk";
}

def Lattigo_RLWEPublicKeyOrSecretKey : AnyTypeOf<[Lattigo_RLWEPublicKey, Lattigo_RLWESecretKey]>;

def Lattigo_RLWERelinearizationKey : Lattigo_RLWEType<"RelinearizationKey", "relinearization_key"> {
  let description = [{
    This type represents the relinearization key for the RLWE encryption scheme.
  }];
  let asmName = "rk";
}

def Lattigo_RLWEGaloisKey : Lattigo_RLWEType<"GaloisKey", "galois_key"> {
  let description = [{
    This type represents the Galois key for the RLWE encryption scheme.

    galoisElement: Enabling the automorphism X -> X^{galoisElement}.
  }];
  let parameters = (ins Builtin_IntegerAttr:$galoisElement);
  let assemblyFormat = "`<` struct(params) `>`";
  let asmName = "gk";
  let aliasSuffix = [{
    os << "_g" << getGaloisElement().getValue().getSExtValue();
  }];
}

def Lattigo_RLWEEvaluationKeySet : Lattigo_RLWEType<"EvaluationKeySet", "evaluation_key_set"> {
  let description = [{
    This type represents the evaluation key set for the RLWE encryption scheme.
  }];
  let asmName = "ekset";
}

def Lattigo_RLWEEncryptor : Lattigo_RLWEType<"Encryptor", "encryptor"> {
  let description = [{
    This type represents the encryptor for the RLWE encryption scheme.
  }];
  let parameters = (ins "bool":$publicKey);
  let assemblyFormat = "`<` struct(params) `>`";
  let asmName = "encryptor";
  let aliasSuffix = [{
    if (getPublicKey()) {
      os<< "_pk";
    } else {
      os<< "_sk";
    }
  }];
}

def Lattigo_RLWEDecryptor : Lattigo_RLWEType<"Decryptor", "decryptor"> {
  let description = [{
    This type represents the decryptor for the RLWE encryption scheme.
  }];
  let asmName = "decryptor";
}

def Lattigo_RLWEPlaintext : Lattigo_RLWEType<"Plaintext", "plaintext"> {
  let description = [{
    This type represents the plaintext for the RLWE encryption scheme.
  }];
  let asmName = "pt";
}

def Lattigo_RLWECiphertext : Lattigo_RLWEType<"Ciphertext", "ciphertext"> {
  let description = [{
    This type represents the ciphertext for the RLWE encryption scheme.
  }];
  // RLWE Ciphertext should have recorded the scaling factor.
  // We rely on Lattigo internal implementation to track the scaling factor.
  let asmName = "ct";
}

def Lattigo_RotatedCiphertextList : Lattigo_RLWEType<"RotatedCiphertextList", "rotated_ciphertext_map"> {
  let description = [{
    This type represents a list of rotated ciphertexts from a bulk rotation op.
  }];
  let asmName = "rotated_ct_map";
}

def Lattigo_RLWECiphertextOrPlaintext : AnyTypeOf<[Lattigo_RLWECiphertext, Lattigo_RLWEPlaintext]>;

// common interface for RLWE
def Lattigo_RLWEParameter : AnyTypeOf<[Lattigo_BGVParameter, Lattigo_CKKSParameter]>;
def Lattigo_RLWEEncoder : AnyTypeOf<[Lattigo_BGVEncoder, Lattigo_CKKSEncoder]>;
def Lattigo_RLWEEvaluator : AnyTypeOf<[Lattigo_BGVEvaluator, Lattigo_CKKSEvaluator]>;

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGORLWETYPES_TD_
