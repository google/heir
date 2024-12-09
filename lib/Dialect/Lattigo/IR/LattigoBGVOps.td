#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOBGVOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOBGVOPS_TD_

include "LattigoDialect.td"
include "LattigoTypes.td"
include "mlir/IR/OpBase.td"

class Lattigo_BGVOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_Op<"bgv." # mnemonic, traits> {
}

def Lattigo_BGVNewPlaintextOp : Lattigo_BGVOp<"new_plaintext"> {
  let summary = "Create a new plaintext in the Lattigo BGV dialect";
  let description = [{
    This operation creates a new plaintext value in the Lattigo BGV dialect.
  }];
  let arguments = (ins
    Lattigo_BGVParameter:$params
  );
  let results = (outs Lattigo_RLWEPlaintext:$plaintext);
}

def Lattigo_BGVNewParametersFromLiteralOp : Lattigo_BGVOp<"new_parameters_from_literal"> {
  let summary = "Create new BGV parameters from a literal in the Lattigo BGV dialect";
  let description = [{
    This operation creates new BGV parameters from a given literal value in the Lattigo BGV dialect.
  }];
  let arguments = (ins
    Lattigo_BGVParametersLiteral:$paramsLiteral
  );
  let results = (outs Lattigo_BGVParameter:$params);
}

def Lattigo_BGVNewEncoderOp : Lattigo_BGVOp<"new_encoder"> {
  let summary = "Create a new encoder in the Lattigo BGV dialect";
  let description = [{
    This operation creates a new encoder for encoding plaintext values in the Lattigo BGV dialect.
  }];
  let arguments = (ins
    Lattigo_BGVParameter:$params
  );
  let results = (outs Lattigo_BGVEncoder:$encoder);
}

def Lattigo_BGVEncodeOp : Lattigo_BGVOp<"encode"> {
  let summary = "Encode a plaintext value in the Lattigo BGV dialect";
  let description = [{
    This operation encodes a plaintext value using the specified encoder in the Lattigo BGV dialect.
  }];
  let arguments = (ins
    Lattigo_BGVEncoder:$encoder,
    AnyType:$value,
    Lattigo_RLWEPlaintext:$plaintext
  );
  let results = (outs Lattigo_RLWEPlaintext:$encoded);

  let hasVerifier = 1;
}

def Lattigo_BGVDecodeOp : Lattigo_BGVOp<"decode"> {
  let summary = "Decode a plaintext value in the Lattigo BGV dialect";
  let description = [{
    This operation decodes a plaintext value using the specified encoder in the Lattigo BGV dialect.
  }];
  let arguments = (ins
    Lattigo_BGVEncoder:$encoder,
    Lattigo_RLWEPlaintext:$plaintext,
    AnyType:$value
  );
  let results = (outs AnyType:$decoded);

  let hasVerifier = 1;
}

def Lattigo_BGVNewEvaluatorOp : Lattigo_BGVOp<"new_evaluator"> {
  let summary = "Create a new evaluator in the Lattigo BGV dialect";
  let description = [{
    This operation creates a new evaluator for performing operations on ciphertexts in the Lattigo BGV dialect.
  }];
  let arguments = (ins
    Lattigo_BGVParameter:$params
  );
  let results = (outs Lattigo_BGVEvaluator:$evaluator);
}

// ciphertext arithmetic op

class Lattigo_BGVBinaryOp<string mnemonic> :
        Lattigo_BGVOp<mnemonic> {
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$lhs,
    Lattigo_RLWECiphertext:$rhs
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_BGVAddOp : Lattigo_BGVBinaryOp<"add"> {
  let summary = "Add two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation adds two ciphertext values in the Lattigo BGV dialect.
  }];
}

def Lattigo_BGVSubOp : Lattigo_BGVBinaryOp<"sub"> {
  let summary = "Subtract two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation subtracts one ciphertext value from another in the Lattigo BGV dialect.
  }];
}

def Lattigo_BGVMulOp : Lattigo_BGVBinaryOp<"mul"> {
  let summary = "Multiply two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation multiplies two ciphertext values in the Lattigo BGV dialect.
  }];
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOBGVOPS_TD_
