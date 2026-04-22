#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_

include "CheddarDialect.td"
include "CheddarTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "lib/Dialect/HEIRInterfaces.td"

class Cheddar_Op<string mnemonic, list<Trait> traits = []> :
        Op<Cheddar_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Setup operations
//===----------------------------------------------------------------------===//

def Cheddar_CreateContextOp : Cheddar_Op<"create_context"> {
  let summary = "Create a CHEDDAR context from parameters";
  let description = [{
    Creates a CHEDDAR Context<word> from a Parameter object.
    The context is the main server-side computation engine.
  }];
  let arguments = (ins Cheddar_Parameter:$params);
  let results = (outs Cheddar_Context:$ctx);
}

def Cheddar_CreateUserInterfaceOp : Cheddar_Op<"create_user_interface"> {
  let summary = "Create a CHEDDAR UserInterface for key gen and encrypt/decrypt";
  let description = [{
    Creates a UserInterface<word> from a context. The UserInterface handles
    key generation, encryption, and decryption. Note: for test purposes only.
  }];
  let arguments = (ins Cheddar_Context:$ctx);
  let results = (outs Cheddar_UserInterface:$ui);
}

def Cheddar_GetEncoderOp : Cheddar_Op<"get_encoder"> {
  let summary = "Get the encoder from a CHEDDAR context";
  let description = [{
    Returns a reference to the context's encoder (context->encoder_).
  }];
  let arguments = (ins Cheddar_Context:$ctx);
  let results = (outs Cheddar_Encoder:$encoder);
}

def Cheddar_GetEvkMapOp : Cheddar_Op<"get_evk_map"> {
  let summary = "Get the evaluation key map from a UserInterface";
  let description = [{
    Returns the EvkMap from the UserInterface (ui.GetEvkMap()).
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvkMap:$evkMap);
}

def Cheddar_GetMultKeyOp : Cheddar_Op<"get_mult_key"> {
  let summary = "Get the multiplication evaluation key";
  let description = [{
    Returns the multiplication key from the UserInterface
    (ui.GetMultiplicationKey()).
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvalKey:$key);
}

def Cheddar_GetRotKeyOp : Cheddar_Op<"get_rot_key"> {
  let summary = "Get a rotation evaluation key";
  let description = [{
    Returns a rotation key for the given distance from the UserInterface
    (ui.GetRotationKey(dist)).
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Builtin_IntegerAttr:$distance
  );
  let results = (outs Cheddar_EvalKey:$key);
}

def Cheddar_GetConjKeyOp : Cheddar_Op<"get_conj_key"> {
  let summary = "Get the conjugation evaluation key";
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvalKey:$key);
}

def Cheddar_PrepareRotKeyOp : Cheddar_Op<"prepare_rot_key"> {
  let summary = "Generate a rotation key for a given distance";
  let description = [{
    Calls ui.PrepareRotationKey(distance, max_level) to generate a rotation key.
    Must be called before using rotation with that distance.
    The max_level parameter specifies the maximum ciphertext level at which
    this rotation key will be used.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Builtin_IntegerAttr:$distance,
    Builtin_IntegerAttr:$maxLevel
  );
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// Encode / Encrypt / Decrypt operations
//===----------------------------------------------------------------------===//

def Cheddar_EncodeOp : Cheddar_Op<"encode"> {
  let summary = "Encode a message vector into a CHEDDAR plaintext";
  let description = [{
    Calls encoder.Encode(pt, level, scale, message). The message is a vector
    of complex numbers (or reals).
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    RankedTensorOf<[AnyFloat, AnyComplex]>:$message,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$scale
  );
  let results = (outs Cheddar_Plaintext:$plaintext);
}

def Cheddar_EncodeConstantOp : Cheddar_Op<"encode_constant"> {
  let summary = "Encode a scalar double into a CHEDDAR constant";
  let description = [{
    Calls encoder.EncodeConstant(constant, level, scale, number).
    The result is in RNS form for efficient ciphertext-scalar ops.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    AnyFloat:$value,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$scale
  );
  let results = (outs Cheddar_Constant:$constant);
}

def Cheddar_DecodeOp : Cheddar_Op<"decode"> {
  let summary = "Decode a CHEDDAR plaintext back to a message vector";
  let description = [{
    Calls encoder.Decode(message, pt). Returns a vector of complex numbers.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    Cheddar_Plaintext:$plaintext
  );
  let results = (outs RankedTensorOf<[AnyFloat, AnyComplex]>:$message);
}

def Cheddar_EncryptOp : Cheddar_Op<"encrypt"> {
  let summary = "Encrypt a plaintext into a ciphertext";
  let description = [{
    Calls ui.Encrypt(ct, pt). Test-only operation.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Cheddar_Plaintext:$plaintext
  );
  let results = (outs Cheddar_Ciphertext:$ciphertext);
}

def Cheddar_DecryptOp : Cheddar_Op<"decrypt"> {
  let summary = "Decrypt a ciphertext into a plaintext";
  let description = [{
    Calls ui.Decrypt(pt, ct). Test-only operation.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Cheddar_Ciphertext:$ciphertext
  );
  let results = (outs Cheddar_Plaintext:$plaintext);
}

//===----------------------------------------------------------------------===//
// Ciphertext-ciphertext arithmetic operations
//===----------------------------------------------------------------------===//

class Cheddar_BinaryCtCtOp<string mnemonic, list<Trait> traits = []>
    : Cheddar_Op<mnemonic, traits> {
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$lhs,
    Cheddar_Ciphertext:$rhs
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_AddOp : Cheddar_BinaryCtCtOp<"add"> {
  let summary = "Add two ciphertexts";
  let description = [{
    Calls context->Add(res, a, b).
  }];
}

def Cheddar_SubOp : Cheddar_BinaryCtCtOp<"sub"> {
  let summary = "Subtract two ciphertexts";
  let description = [{
    Calls context->Sub(res, a, b).
  }];
}

def Cheddar_MultOp : Cheddar_BinaryCtCtOp<"mult", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts (tensor product, no relin/rescale)";
  let description = [{
    Calls context->Mult(res, a, b). Produces a degree-3 ciphertext.
    Does NOT include relinearization or rescaling.
  }];
}

//===----------------------------------------------------------------------===//
// Ciphertext-plaintext / ciphertext-constant operations
//===----------------------------------------------------------------------===//

def Cheddar_AddPlainOp : Cheddar_Op<"add_plain"> {
  let summary = "Add a plaintext to a ciphertext";
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$ciphertext,
    Cheddar_CiphertextOrPlaintext:$plaintext
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_SubPlainOp : Cheddar_Op<"sub_plain"> {
  let summary = "Subtract a plaintext from a ciphertext";
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$ciphertext,
    Cheddar_CiphertextOrPlaintext:$plaintext
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_MultPlainOp : Cheddar_Op<"mult_plain"> {
  let summary = "Multiply a ciphertext by a plaintext (no rescale)";
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$ciphertext,
    Cheddar_CiphertextOrPlaintext:$plaintext
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_AddConstOp : Cheddar_Op<"add_const"> {
  let summary = "Add a constant to a ciphertext";
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$ciphertext,
    Cheddar_Constant:$constant
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_MultConstOp : Cheddar_Op<"mult_const"> {
  let summary = "Multiply a ciphertext by a constant (no rescale)";
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$ciphertext,
    Cheddar_Constant:$constant
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

//===----------------------------------------------------------------------===//
// Unary ciphertext operations
//===----------------------------------------------------------------------===//

def Cheddar_NegOp : Cheddar_Op<"neg"> {
  let summary = "Negate a ciphertext";
  let description = [{
    Calls context->Neg(res, a).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_RescaleOp : Cheddar_Op<"rescale"> {
  let summary = "Rescale a ciphertext (drop one level)";
  let description = [{
    Calls context->Rescale(res, a). Reduces the ciphertext level by 1.
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_LevelDownOp : Cheddar_Op<"level_down"> {
  let summary = "Reduce ciphertext to a target level";
  let description = [{
    Calls context->LevelDown(res, a, target_level).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Builtin_IntegerAttr:$targetLevel
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

//===----------------------------------------------------------------------===//
// Key-switching operations
//===----------------------------------------------------------------------===//

def Cheddar_RelinearizeOp : Cheddar_Op<"relinearize"> {
  let summary = "Relinearize a ciphertext (without rescale)";
  let description = [{
    Calls context->Relinearize(res, a, mult_key).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvalKey:$multKey
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_RelinearizeRescaleOp : Cheddar_Op<"relinearize_rescale"> {
  let summary = "Fused relinearize + rescale";
  let description = [{
    Calls context->RelinearizeRescale(res, a, mult_key). Faster than
    separate Relinearize + Rescale.
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvalKey:$multKey
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

//===----------------------------------------------------------------------===//
// Compound (fused) operations -- high-performance GPU kernels
//===----------------------------------------------------------------------===//

def Cheddar_HMultOp : Cheddar_Op<"hmult", [IncreasesMulDepthOpInterface]> {
  let summary = "Fused multiply + relinearize (+ optional rescale)";
  let description = [{
    Calls context->HMult(res, a, b, mult_key, rescale).
    Single fused GPU kernel launch. The `rescale` attribute controls whether
    rescaling is included (default: true).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$lhs,
    Cheddar_Ciphertext:$rhs,
    Cheddar_EvalKey:$multKey,
    DefaultValuedAttr<BoolAttr, "true">:$rescale
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_HRotOp : Cheddar_Op<"hrot", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Fused key-switch + rotation";
  let description = [{
    Calls context->HRot(res, a, rot_key, distance).
    Single fused GPU kernel for rotation.
    Supports both static and dynamic shift distances.
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvalKey:$rotKey,
    Optional<AnySignlessIntegerOrIndex>:$dynamic_shift,
    OptionalAttr<Builtin_IntegerAttr>:$static_shift
  );
  let results = (outs Cheddar_Ciphertext:$output);
  let hasVerifier = 1;
}

def Cheddar_HRotAddOp : Cheddar_Op<"hrot_add", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Fused rotation + addition";
  let description = [{
    Computes res = rotate(a, distance) + b in a single fused GPU kernel.
    Calls context->HRotAdd(res, a, b, rot_key, distance).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_Ciphertext:$addend,
    Cheddar_EvalKey:$rotKey,
    Builtin_IntegerAttr:$distance
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_HConjOp : Cheddar_Op<"hconj"> {
  let summary = "Fused key-switch + conjugation";
  let description = [{
    Calls context->HConj(res, a, conj_key).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvalKey:$conjKey
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_HConjAddOp : Cheddar_Op<"hconj_add"> {
  let summary = "Fused conjugation + addition";
  let description = [{
    Computes res = conj(a) + b in a single fused GPU kernel.
    Calls context->HConjAdd(res, a, b, conj_key).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_Ciphertext:$addend,
    Cheddar_EvalKey:$conjKey
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_MadUnsafeOp : Cheddar_Op<"mad_unsafe"> {
  let summary = "Fused multiply-accumulate with constant (no rescale)";
  let description = [{
    Computes res += a * constant (in-place accumulation).
    Calls context->MadUnsafe(res, a, constant).
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$accumulator,
    Cheddar_Ciphertext:$input,
    Cheddar_Constant:$constant
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

//===----------------------------------------------------------------------===//
// Extension operations (bootstrapping, linear transforms, poly eval)
//===----------------------------------------------------------------------===//

def Cheddar_BootOp : Cheddar_Op<"boot", [ResetsMulDepthOpInterface]> {
  let summary = "Bootstrap a ciphertext";
  let description = [{
    Calls boot_ctx->Boot(res, input, evk_map).
    Refreshes the ciphertext noise budget.
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvkMap:$evkMap
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_LinearTransformOp : Cheddar_Op<"linear_transform", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Apply a linear transform on a ciphertext";
  let description = [{
    Applies a matrix-vector product using CHEDDAR's LinearTransform extension
    with BSGS optimization and hoisting.

    The `diagonals` input is a 2D tensor where each row is a non-zero diagonal.
    The `diagonal_indices` attribute specifies which diagonal each row represents.
    The `level` attribute specifies the modulus level for the operation.
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvkMap:$evkMap,
    2DTensorOf<[AnyFloat]>:$diagonals,
    DenseI32ArrayAttr:$diagonal_indices,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$logBabyStepGiantStepRatio
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

def Cheddar_EvalPolyOp : Cheddar_Op<"eval_poly"> {
  let summary = "Evaluate a polynomial on a ciphertext";
  let description = [{
    Evaluates a polynomial (e.g., Chebyshev approximation) on an encrypted
    input using CHEDDAR's EvalPoly extension.
  }];
  let arguments = (ins
    Cheddar_Context:$ctx,
    Cheddar_Ciphertext:$input,
    Cheddar_EvkMap:$evkMap,
    ArrayAttr:$coefficients,
    Builtin_IntegerAttr:$level
  );
  let results = (outs Cheddar_Ciphertext:$output);
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_
