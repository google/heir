#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOBGVOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOBGVOPS_TD_

include "LattigoDialect.td"
include "LattigoTypes.td"
include "lib/Dialect/HEIRInterfaces.td"
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

def Lattigo_BGVEncodeOp : Lattigo_BGVOp<"encode", [InPlaceOpInterface]> {
  let summary = "Encode a plaintext value in the Lattigo BGV dialect";
  let description = [{
    This operation encodes a plaintext value using the specified encoder in the Lattigo BGV dialect.

    The input value tensor should have elements of integer type.

    The `scale` attribute indicates the scaling factor for encoding.
  }];
  let arguments = (ins
    Lattigo_BGVEncoder:$encoder,
    RankedTensorOf<[AnyInteger]>:$value,
    Lattigo_RLWEPlaintext:$plaintext,
    DefaultValuedAttr<I64Attr, "1">:$scale
  );
  let results = (outs Lattigo_RLWEPlaintext:$encoded);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 2; }";
}

def Lattigo_BGVDecodeOp : Lattigo_BGVOp<"decode", [AllTypesMatch<["value", "decoded"]>]> {
  let summary = "Decode a plaintext value in the Lattigo BGV dialect";
  let description = [{
    This operation decodes a plaintext value using the specified encoder in the Lattigo BGV dialect.

    The input value tensor should have elements of integer type.
  }];
  let arguments = (ins
    Lattigo_BGVEncoder:$encoder,
    Lattigo_RLWEPlaintext:$plaintext,
    RankedTensorOf<[AnyInteger]>:$value
    // Rely on Lattigo internal tracking of scale for decoding.
    // DefaultValuedAttr<I64Attr, "1">:$scale
  );
  // although bgv.Decode is also an inplace operation as bgv.Encode, as there are post-processing
  // steps in emitter, we mark it as a normal operation.
  let results = (outs RankedTensorOf<[AnyInteger]>:$decoded);
}

def Lattigo_BGVNewEvaluatorOp : Lattigo_BGVOp<"new_evaluator"> {
  let summary = "Create a new evaluator in the Lattigo BGV dialect";
  let description = [{
    This operation creates a new evaluator for performing operations on ciphertexts in the Lattigo BGV dialect.

    By default, the evaluator is created with the provided parameters and could execute
    operations which does not relying on evaluation keys.

    To support operations that require evaluation keys,
    the optional evaluation key set should be provided.

    The scaleInvariant flag is used to indicate whether the evaluator is for B/FV or BGV.
    If it is set to true, the evaluator will evaluate operations in B/FV style.
  }];
  let arguments = (ins
    Lattigo_BGVParameter:$params,
    Optional<Lattigo_RLWEEvaluationKeySet>:$evaluationKeySet,
    DefaultValuedAttr<BoolAttr, "false">:$scaleInvariant
  );
  let results = (outs Lattigo_BGVEvaluator:$evaluator);
}

// ciphertext arithmetic op

class Lattigo_BGVBinaryOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_BGVOp<mnemonic, traits> {
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$lhs,
    Lattigo_RLWECiphertextOrPlaintext:$rhs
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_BGVAddNewOp : Lattigo_BGVBinaryOp<"add_new"> {
  let summary = "Add two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation adds two ciphertext values in the Lattigo BGV dialect.
  }];
}

def Lattigo_BGVSubNewOp : Lattigo_BGVBinaryOp<"sub_new"> {
  let summary = "Subtract two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation subtracts one ciphertext value from another in the Lattigo BGV dialect.
  }];
}

def Lattigo_BGVMulNewOp : Lattigo_BGVBinaryOp<"mul_new", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation multiplies two ciphertext values in the Lattigo BGV dialect.
  }];
}

class Lattigo_BGVBinaryInPlaceOp<string mnemonic, list<Trait> traits = []> :
        Lattigo_BGVOp<mnemonic, traits # [InPlaceOpInterface]> {
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$lhs,
    Lattigo_RLWECiphertextOrPlaintext:$rhs,
    // Lattigo API is like bgv.Add(lhs, rhs, out) but for MLIR we need to
    // satisfy the SSA form, so we still have a separate output.
    Lattigo_RLWECiphertext:$inplace
  );
  let results = (outs Lattigo_RLWECiphertext:$output);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 3; }";
}

def Lattigo_BGVAddOp : Lattigo_BGVBinaryInPlaceOp<"add"> {
  let summary = "Add two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation adds two ciphertext values in the Lattigo BGV dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_BGVSubOp : Lattigo_BGVBinaryInPlaceOp<"sub"> {
  let summary = "Subtract two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation subtracts one ciphertext value from another in the Lattigo BGV dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_BGVMulOp : Lattigo_BGVBinaryInPlaceOp<"mul", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts in the Lattigo BGV dialect";
  let description = [{
    This operation multiplies two ciphertext values in the Lattigo BGV dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

class Lattigo_BGVUnaryOp<string mnemonic> :
        Lattigo_BGVOp<mnemonic> {
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_BGVRelinearizeNewOp : Lattigo_BGVUnaryOp<"relinearize_new"> {
  let summary = "Relinearize a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation relinearizes a ciphertext value in the Lattigo BGV dialect.
  }];
}

def Lattigo_BGVRescaleNewOp : Lattigo_BGVUnaryOp<"rescale_new"> {
  let summary = "Rescale a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation rescales a ciphertext value in the Lattigo BGV dialect.
  }];
}

def Lattigo_BGVRotateColumnsNewOp : Lattigo_BGVOp<"rotate_columns_new"> {
  let summary = "Rotate columns of a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation rotates the columns of a ciphertext value in the Lattigo BGV dialect.

    Lattigo exposes the SIMD slot of BGV as a N/2 x 2 matrix, where N/2 is the column.

    Offset is valid for both positive and negative number.
  }];
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    Optional<I32>:$shift,
    OptionalAttr<Builtin_IntegerAttr>:$offset
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_BGVRotateRowsNewOp : Lattigo_BGVUnaryOp<"rotate_rows_new"> {
  let summary = "Rotate rows of a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation swap the rows of a ciphertext value in the Lattigo BGV dialect.

    Lattigo exposes the SIMD slot of BGV as a N/2 x 2 matrix, where 2 is the row.
  }];
}

class Lattigo_BGVUnaryInPlaceOp<string mnemonic> :
        Lattigo_BGVOp<mnemonic, [InPlaceOpInterface]> {
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    // see BinaryInPlaceOp above
    Lattigo_RLWECiphertext:$inplace
  );
  let results = (outs Lattigo_RLWECiphertext:$output);

  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 2; }";
}

def Lattigo_BGVRelinearizeOp : Lattigo_BGVUnaryInPlaceOp<"relinearize"> {
  let summary = "Relinearize a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation relinearizes a ciphertext value in the Lattigo BGV dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_BGVRescaleOp : Lattigo_BGVUnaryInPlaceOp<"rescale"> {
  let summary = "Rescale a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation rescales a ciphertext value in the Lattigo BGV dialect.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

def Lattigo_BGVRotateColumnsOp : Lattigo_BGVUnaryInPlaceOp<"rotate_columns"> {
  let summary = "Rotate columns of a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation rotates the columns of a ciphertext value in the Lattigo BGV dialect.

    Lattigo exposes the SIMD slot of BGV as a N/2 x 2 matrix, where N/2 is the column.

    Offset is valid for both positive and negative number.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
  let arguments = (ins
    Lattigo_BGVEvaluator:$evaluator,
    Lattigo_RLWECiphertext:$input,
    Lattigo_RLWECiphertext:$inplace,
    Optional<I32>:$shift,
    OptionalAttr<Builtin_IntegerAttr>:$offset
  );
  let results = (outs Lattigo_RLWECiphertext:$output);
}

def Lattigo_BGVRotateRowsOp : Lattigo_BGVUnaryInPlaceOp<"rotate_rows"> {
  let summary = "Rotate rows of a ciphertext in the Lattigo BGV dialect";
  let description = [{
    This operation swap the rows of a ciphertext value in the Lattigo BGV dialect.

    Lattigo exposes the SIMD slot of BGV as a N/2 x 2 matrix, where 2 is the row.

    The result will be written to the `inplace` operand. The `output`result is
    a transitive reference to the `inplace` operand for sake of the MLIR SSA form.
  }];
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOBGVOPS_TD_
