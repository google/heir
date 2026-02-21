#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGORLWEOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGORLWEOPS_TD_

class Lattigo_RLWEOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_Op<"rlwe." # mnemonic, traits> {
}

def Lattigo_RLWENewKeyGeneratorOp : Lattigo_RLWEOp<"new_key_generator"> {
  let summary = "Generates a new RLWE key generator";
  let description = [{
    This operation generates a new RLWE key generator
  }];
  let arguments = (ins
    Lattigo_RLWEParameter:$params
  );
  let results = (outs Lattigo_RLWEKeyGenerator:$keyGenerator);
}

def Lattigo_RLWEGenKeyPairOp : Lattigo_RLWEOp<"gen_key_pair"> {
  let summary = "Generates a new RLWE key pair";
  let description = [{
    This operation generates a new RLWE key pair
  }];
  let arguments = (ins
    Lattigo_RLWEKeyGenerator:$keyGenerator
  );
  let results = (outs
    Lattigo_RLWESecretKey:$secretKey,
    Lattigo_RLWEPublicKey:$publicKey
  );
}

def Lattigo_RLWEGenRelinearizationKeyOp : Lattigo_RLWEOp<"gen_relinearization_key"> {
  let summary = "Generates a new RLWE relinearization key";
  let description = [{
    This operation generates a new RLWE relinearization key
  }];
  let arguments = (ins
    Lattigo_RLWEKeyGenerator:$keyGenerator,
    Lattigo_RLWESecretKey:$secretKey
  );
  let results = (outs Lattigo_RLWERelinearizationKey:$relinearizationKey);
}

def Lattigo_RLWEGenGaloisKeyOp : Lattigo_RLWEOp<"gen_galois_key"> {
  let summary = "Generates a new RLWE Galois key";
  let description = [{
    This operation generates a new RLWE Galois key

    galoisElement: Enabling the automorphism X -> X^{galoisElement}.
  }];
  let arguments = (ins
    Lattigo_RLWEKeyGenerator:$keyGenerator,
    Lattigo_RLWESecretKey:$secretKey,
    Builtin_IntegerAttr:$galoisElement
  );
  let results = (outs Lattigo_RLWEGaloisKey:$galoisKey);
}

def Lattigo_RLWENewEvaluationKeySetOp : Lattigo_RLWEOp<"new_evaluation_key_set"> {
  let summary = "Generates a new RLWE evaluation key set";
  let description = [{
    This operation generates a new RLWE evaluation key set
  }];

  // Tablegen produces an error when validating types when using a TypeOrContainer
  // type in a Variadic like Variadic<AnyEvaluationKey>.
  // A workaround uses Variadic<AnyType> and a custom type verifier.
  let arguments = (ins
    Variadic<AnyType>:$keys
  );
  let results = (outs Lattigo_RLWEEvaluationKeySet:$evaluationKeySet);
  let hasVerifier = 1;
}

def Lattigo_RLWENewEncryptorOp : Lattigo_RLWEOp<"new_encryptor"> {
  let summary = "Creates a new RLWE encryptor";
  let description = [{
    This operation creates a new RLWE encryptor
  }];
  let arguments = (ins
    Lattigo_RLWEParameter:$params,
    Lattigo_RLWEPublicKeyOrSecretKey:$encryptionKey
  );
  let results = (outs Lattigo_RLWEEncryptor:$encryptor);
  let hasVerifier = 1;
}

def Lattigo_RLWENewDecryptorOp : Lattigo_RLWEOp<"new_decryptor"> {
  let summary = "Creates a new RLWE decryptor";
  let description = [{
    This operation creates a new RLWE decryptor
  }];
  let arguments = (ins
    Lattigo_RLWEParameter:$params,
    Lattigo_RLWESecretKey:$secretKey
  );
  let results = (outs Lattigo_RLWEDecryptor:$decryptor);
}

def Lattigo_RLWEEncryptOp : Lattigo_RLWEOp<"encrypt"> {
  let summary = "Encrypts a plaintext using RLWE";
  let description = [{
    This operation encrypts a plaintext using RLWE
  }];
  let arguments = (ins
    Lattigo_RLWEEncryptor:$encryptor,
    Lattigo_RLWEPlaintext:$plaintext
  );
  let results = (outs Lattigo_RLWECiphertext:$ciphertext);
}

def Lattigo_RLWEDecryptOp : Lattigo_RLWEOp<"decrypt"> {
  let summary = "Decrypts a ciphertext using RLWE";
  let description = [{
    This operation decrypts a ciphertext using RLWE
  }];
  let arguments = (ins
    Lattigo_RLWEDecryptor:$decryptor,
    Lattigo_RLWECiphertext:$ciphertext
  );
  let results = (outs Lattigo_RLWEPlaintext:$plaintext);
}

def Lattigo_RLWEDropLevelNewOp : Lattigo_RLWEOp<"drop_level_new", 
  [DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop"]>]> {
  let summary = "Drop level of a ciphertext";
  let arguments = (ins
    Lattigo_RLWEEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_RLWEDropLevelOp : Lattigo_RLWEOp<"drop_level", 
  [InPlaceOpInterface, DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop"]>]> {
  let summary = "Drop level of a ciphertext";
  let description = [{
    This operation drops the level of a ciphertext

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
  let arguments = (ins
    Lattigo_RLWEEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    Lattigo_RLWECiphertext:$inplace,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
  let results = (outs Lattigo_RLWECiphertext:$output);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 2; }";
}

def Lattigo_RLWENegateNewOp : Lattigo_RLWEOp<"negate_new"> {
  let summary = "Negate a ciphertext";
  let arguments = (ins
    Lattigo_RLWEEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_RLWENegateOp : Lattigo_RLWEOp<"negate", [InPlaceOpInterface]> {
  let summary = "Negate of a ciphertext";
  let description = [{
    This operation negates a ciphertext

    The result will be written to the `inplace` operand. The `output` result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
  let arguments = (ins
    Lattigo_RLWEEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    Lattigo_RLWECiphertext:$inplace
  );
  let results = (outs Lattigo_RLWECiphertext:$output);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 2; }";
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGORLWEOPS_TD_
