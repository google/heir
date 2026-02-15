#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSOPS_TD_

include "LattigoDialect.td"
include "LattigoTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "lib/Dialect/HEIRInterfaces.td"
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

def Lattigo_CKKSEncodeOp : Lattigo_CKKSOp<"encode", [InPlaceOpInterface]> {
  let summary = "Encode a plaintext value in the Lattigo CKKS dialect";
  let description = [{
    This operation encodes a plaintext value using the specified encoder in the Lattigo CKKS dialect.

    The input value tensor should have elements of float type or complex type.

    The `scale` attribute indicates the scaling factor for encoding.
  }];
  let arguments = (ins
    Lattigo_CKKSEncoder:$encoder,
    RankedTensorOf<[AnyFloat, AnyComplex]>:$value,
    Lattigo_RLWEPlaintext:$plaintext,
    DefaultValuedAttr<I64Attr, "1">:$scale
  );
  let results = (outs Lattigo_RLWEPlaintext:$encoded);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 2; }";
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
  // although bgv.Decode is also an inplace operation as bgv.Encode, as there are post-processing
  // steps in emitter, we mark it as a normal operation.
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

def Lattigo_CKKSNewBootstrappingParametersFromLiteralOp : Lattigo_CKKSOp<"new_bootstrapping_parameters_from_literal"> {
  let summary = "Create new CKKS bootstrapping parameters from a literal in the Lattigo CKKS dialect";
  let description = [{
    This operation creates new CKKS bootstrapping parameters from a given literal value in the Lattigo CKKS dialect.
  }];
  let arguments = (ins
    Lattigo_CKKSParameter:$params,
    Lattigo_CKKSBootstrappingParametersLiteral:$btParamsLiteral
  );
  let results = (outs Lattigo_CKKSBootstrappingParameter:$output);
}

def Lattigo_CKKSGenEvaluationKeysBootstrappingOp : Lattigo_CKKSOp<"gen_evaluation_keys_bootstrapping"> {
  let summary = "Create new evaluation keys for CKKS bootstrapping";
  let arguments = (ins
    Lattigo_CKKSBootstrappingParameter:$params,
    Lattigo_RLWESecretKey:$sk
  );
  let results = (outs Lattigo_CKKSBootstrappingEvaluationKeys:$output);
}

def Lattigo_CKKSNewBootstrappingEvaluatorOp : Lattigo_CKKSOp<"new_bootstrapping_evaluator"> {
  let summary = "Create a new bootstrapping evaluator in the Lattigo CKKS dialect";
  let description = [{
    Creates a new bootstrapping evaluator for performing operations on
    ciphertexts in the Lattigo CKKS dialect.

    By default, the evaluator is created with default parameters to provide a
    depth of 15 and security level of 128. The evaluation key set for
    bootstrapping must be provided.
  }];
  let arguments = (ins
    Lattigo_CKKSBootstrappingParameter:$params,
    Lattigo_CKKSBootstrappingEvaluationKeys:$evaluationKeySet
  );
  let results = (outs Lattigo_CKKSBootstrappingEvaluator:$evaluator);
}

// ciphertext arithmetic op

class Lattigo_CKKSBinaryOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_CKKSOp<mnemonic, traits> {
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$lhs,
    Lattigo_RLWECiphertextOrPlaintext:$rhs
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_CKKSAddNewOp : Lattigo_CKKSBinaryOp<"add_new"> {
  let summary = "Add two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation adds two ciphertext values in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSSubNewOp : Lattigo_CKKSBinaryOp<"sub_new"> {
  let summary = "Subtract two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation subtracts one ciphertext value from another in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSMulNewOp : Lattigo_CKKSBinaryOp<"mul_new", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation multiplies two ciphertext values in the Lattigo CKKS dialect.
  }];
}

class Lattigo_CKKSBinaryInPlaceOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_CKKSOp<mnemonic, traits # [InPlaceOpInterface]> {
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$lhs,
    Lattigo_RLWECiphertextOrPlaintext:$rhs,
    // Lattigo API is like bgv.Add(lhs, rhs, out) but for MLIR we need to
    // satisfy the SSA form, so we still have a separate output.
    Lattigo_RLWECiphertext:$inplace
  );
  let results = (outs Lattigo_RLWECiphertext:$output);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 3; }";
}

def Lattigo_CKKSAddOp : Lattigo_CKKSBinaryInPlaceOp<"add"> {
  let summary = "Add two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation adds two ciphertext values in the Lattigo CKKS dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_CKKSSubOp : Lattigo_CKKSBinaryInPlaceOp<"sub"> {
  let summary = "Subtract two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation subtracts one ciphertext value from another in the Lattigo CKKS dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_CKKSMulOp : Lattigo_CKKSBinaryInPlaceOp<"mul", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts in the Lattigo CKKS dialect";
  let description = [{
    This operation multiplies two ciphertext values in the Lattigo CKKS dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
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

def Lattigo_CKKSRelinearizeNewOp : Lattigo_CKKSUnaryOp<"relinearize_new"> {
  let summary = "Relinearize a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation relinearizes a ciphertext value in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSRescaleNewOp : Lattigo_CKKSUnaryOp<"rescale_new"> {
  let summary = "Rescale a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation rescales a ciphertext value in the Lattigo CKKS dialect.
  }];
}

def Lattigo_CKKSRotateNewOp : Lattigo_CKKSOp<"rotate_new"> {
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
    Optional<I32>:$shift,
    OptionalAttr<Builtin_IntegerAttr>:$offset
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

class Lattigo_CKKSUnaryInPlaceOp<string mnemonic> :
        Lattigo_CKKSOp<mnemonic, [InPlaceOpInterface]> {
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    // see BinaryInPlaceOp above
    Lattigo_RLWECiphertext:$inplace
  );
  let results = (outs Lattigo_RLWECiphertext:$output);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 2; }";
}

def Lattigo_CKKSRelinearizeOp : Lattigo_CKKSUnaryInPlaceOp<"relinearize"> {
  let summary = "Relinearize a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation relinearizes a ciphertext value in the Lattigo CKKS dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_CKKSRescaleOp : Lattigo_CKKSUnaryInPlaceOp<"rescale"> {
  let summary = "Rescale a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation rescales a ciphertext value in the Lattigo CKKS dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_CKKSRotateOp : Lattigo_CKKSUnaryInPlaceOp<"rotate"> {
  let summary = "Rotate slots of a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    This operation rotates slots of a ciphertext value in the Lattigo CKKS dialect.

    For vanilla CKKS, the maximum number of slots is N/2 with each slot being complex number.
    Lattigo also support a conjugate-invariant version of CKKS, i.e. the ring is
    Z[X + X^{-1} ]/(X^N+1), which allows for a maximum of N slots with each slot being real number.

    Offset is valid for both positive and negative number.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    // see BinaryInPlaceOp above
    Lattigo_RLWECiphertext:$inplace,
    Optional<I32>:$shift,
    OptionalAttr<Builtin_IntegerAttr>:$offset
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_CKKSBootstrapOp : Lattigo_CKKSUnaryOp<"bootstrap"> {
  let summary = "Bootstrap a ciphertext in the Lattigo CKKS dialect";
  let description = [{
    Bootstraps a ciphertext value in the Lattigo CKKS dialect.

    The operation applies bootstrapping in-place and also returns the result.

    It takes a ciphertext at level 0 (if not at level 0, then it will reduce it
    to level 0) and returns a ciphertext with the max level of
    `evaluator.ResidualParameters.MaxLevel`.
  }];

  let arguments = (ins
    Lattigo_CKKSBootstrappingEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_CKKSLinearTransformOp : Lattigo_CKKSOp<"linear_transform"> {
  let summary = "Apply a linear transform on a lattigo CKKS ciphertext";
  let description = [{
    This operation applies a linear transform on a CKKS ciphertext using
    the provided float diagonals.

    The linear transform is defined by a set of diagonals, where each diagonal
    represents a specific shift and scaling of the input ciphertext slots.

    The `diagonals` input is a 2D tensor where each row represents one non-zero
    diagonal of the square matrix to evaluate. The diagonal values are floats
    that will be encoded into plaintexts during code generation.

    The `levelQ` attribute specifies the modulus level at which the operation
    should be performed.

    The `logBabyStepGiantStepRatio` attribute is used to optimize the linear
    transformation using the baby-step giant-step algorithm. It defines the
    ratio between the sizes of the baby steps and giant steps. If unset,
    it is zero by default.

    During code generation, this op will:
    1. Create a lintrans.Diagonals map from the input tensor
    2. Create and encode a lintrans.Transformation
    3. Create a lintrans.Evaluator
    4. Evaluate the transformation on the input ciphertext
  }];
  let arguments = (ins
    Lattigo_CKKSEvaluator:$evaluator,
    Lattigo_CKKSEncoder:$encoder,
    Lattigo_RLWECiphertext:$input,

    // Parameters corresponding to the lattigo lintrans.Parameters struct
    2DTensorOf<[AnyFloat]>:$diagonals,
    DenseI32ArrayAttr:$diagonal_indices,
    Builtin_IntegerAttr:$levelQ,
    Builtin_IntegerAttr:$logBabyStepGiantStepRatio

    // The remaining parameters from lintrans.Parameters can be inferred from
    // the IR are commented out, but in case we need to expose them in the
    // future, they are listed here.

    // The same auxiliary prime used to generate evaluation keys
    // Builtin_IntegerAttr:$levelP,

    // For CKKS these are hard-coded to [1, N/2] for N = ring modulus degree
    // Builtin_IntegerAttr:$logDimensionsRows,
    // Builtin_IntegerAttr:$logDimensionsCols,
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_CKKSNewPolynomialEvaluatorOp : Lattigo_CKKSOp<"new_polynomial_evaluator"> {
  let summary = "Create a new polynomial evaluator in the Lattigo CKKS dialect";
  let description = [{
    This operation creates a new evaluator for evaluating polynomials in the
    Lattigo CKKS dialect.
  }];
  let arguments = (ins
    Lattigo_CKKSParameter:$params,
    Lattigo_CKKSEvaluator:$evaluator
  );
  let results = (outs Lattigo_CKKSPolynomialEvaluator:$polynomialEvaluator);
}


def Lattigo_CKKSChebyshevOp : Lattigo_CKKSOp<"chebyshev", [AllTypesMatch<["ciphertext", "output"]>]> {
  let summary = "Evaluate a chebyshev polynomial on a lattigo CKKS ciphertext";
  let description = [{
    This operation evaluates a chebyshev polynomial on a CKKS ciphertext using
    the Lattigo polynomial evaluator API.

    The codegen constructs the Lattigo polynomial and, using an existing polynomial
    evaluator, calls the evaluation routine with a given target scale.
  }];
  let arguments = (ins
    Lattigo_CKKSPolynomialEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$ciphertext,
    ArrayAttr:$coefficients,
    Builtin_IntegerAttr:$targetScale
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}


#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSOPS_TD_
