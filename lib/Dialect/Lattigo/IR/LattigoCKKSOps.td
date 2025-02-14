#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSOPS_TD_

include "LattigoDialect.td"
include "LattigoTypes.td"
include "mlir/IR/OpBase.td"

class Lattigo_CKKSOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_Op<"ckks." # mnemonic, traits> {
}

def Lattigo_CKKSNewPlaintextOp : Lattigo_CKKSOp<"new_plaintext"> {
  let summary = "Create a new plaintext in the Lattigo CKKS dialect";
  let description = [{
    This operation creates a new plaintext value in the Lattigo CKKS dialect.
  }];
  let arguments = (ins
    Lattigo_CKKSParameter:$params
  );
  let results = (outs Lattigo_RLWEPlaintext:$plaintext);
}

def Lattigo_CKKSNewParametersFromLiteralOp : Lattigo_CKKSOp<"new_parameters_from_literal"> {
  let summary = "Create new CKKS parameters from a literal in the Lattigo CKKS dialect";
  let description = [{
    This operation creates new CKKS parameters from a given literal value in the Lattigo CKKS dialect.
  }];
  let arguments = (ins
    Lattigo_CKKSParametersLiteral:$paramsLiteral
  );
  let results = (outs Lattigo_CKKSParameter:$params);
}

def Lattigo_CKKSNewEncoderOp : Lattigo_CKKSOp<"new_encoder"> {
  let summary = "Create a new encoder in the Lattigo CKKS dialect";
  let description = [{
    This operation creates a new encoder for encoding plaintext values in the Lattigo CKKS dialect.
  }];
  let arguments = (ins
    Lattigo_CKKSParameter:$params
  );
  let results = (outs Lattigo_CKKSEncoder:$encoder);
}

def Lattigo_CKKSEncodeOp : Lattigo_CKKSOp<"encode"> {
  let summary = "Encode a plaintext value in the Lattigo CKKS dialect";
  let description = [{
    This operation encodes a plaintext value using the specified encoder in the Lattigo CKKS dialect.

    The input value tensor should have elements of float type or complex type.
  }];
  let arguments = (ins
    Lattigo_CKKSEncoder:$encoder,
    RankedTensorOf<[AnyFloat, AnyComplex]>:$value,
    Lattigo_RLWEPlaintext:$plaintext
  );
  let results = (outs Lattigo_RLWEPlaintext:$encoded);
}

def Lattigo_CKKSDecodeOp : Lattigo_CKKSOp<"decode", [AllTypesMatch<["value", "decoded"]>]> {
  let summary = "Decode a plaintext value in the Lattigo CKKS dialect";
  let description = [{
    This operation decodes a plaintext value using the specified encoder in the Lattigo CKKS dialect.

    The input value tensor should have elements of float type or complex type.
  }];
  let arguments = (ins
    Lattigo_CKKSEncoder:$encoder,
    Lattigo_RLWEPlaintext:$plaintext,
    RankedTensorOf<[AnyFloat, AnyComplex]>:$value
  );
  let results = (outs RankedTensorOf<[AnyFloat, AnyComplex]>:$decoded);
}

def Lattigo_CKKSNewEvaluatorOp : Lattigo_CKKSOp<"new_evaluator"> {
  let summary = "Create a new evaluator in the Lattigo CKKS dialect";
  let description = [{
    This operation creates a new evaluator for performing operations on ciphertexts in the Lattigo CKKS dialect.

    By default, the evaluator is created with the provided parameters and could execute
    operations which does not relying on evaluation keys.

    To support operations that require evaluation keys,
    the optional evaluation key set should be provided.
  }];
  let arguments = (ins
    Lattigo_CKKSParameter:$params,
    Optional<Lattigo_RLWEEvaluationKeySet>:$evaluationKeySet
  );
  let results = (outs Lattigo_CKKSEvaluator:$evaluator);
}

// ciphertext arithmetic op

class Lattigo_CKKSBinaryOp<string mnemonic> :
        Lattigo_CKKSOp<mnemonic> {
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$lhs,
    Lattigo_RLWECiphertextOrPlaintext:$rhs
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_CKKSAddOp : Lattigo_CKKSBinaryOp<"add"> {
  let summary = "Add two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation adds two ciphertext values in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSSubOp : Lattigo_CKKSBinaryOp<"sub"> {
  let summary = "Subtract two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation subtracts one ciphertext value from another in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSMulOp : Lattigo_CKKSBinaryOp<"mul"> {
  let summary = "Multiply two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation multiplies two ciphertext values in the Lattigo CKKS dialect.
  }];
}

class Lattigo_CKKSUnaryOp<string mnemonic> :
        Lattigo_CKKSOp<mnemonic> {
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_CKKSRelinearizeOp : Lattigo_CKKSUnaryOp<"relinearize"> {
  let summary = "Relinearize a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation relinearizes a ciphertext value in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSRescaleOp : Lattigo_CKKSUnaryOp<"rescale"> {
  let summary = "Rescale a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation rescales a ciphertext value in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSRotateOp : Lattigo_CKKSOp<"rotate"> {
  let summary = "Rotate slots of a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation rotates slots of a ciphertext value in the Lattigo CKKS dialect.

    For vanilla CKKS, the maximum number of slots is N/2 with each slot being complex number.
    Lattigo also support a conjugate-invariant version of CKKS, i.e. the ring is
    Z[X + X^{-1} ]/(X^N+1), which allows for a maximum of N slots with each slot being real number.

    Offset is valid for both positive and negative number.
  }];
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    Builtin_IntegerAttr:$offset
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSOPS_TD_
