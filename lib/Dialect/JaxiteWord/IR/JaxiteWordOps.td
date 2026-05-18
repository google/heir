#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_TD_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_TD_

include "JaxiteWordDialect.td"
include "JaxiteWordTypes.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWETraits.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class JaxiteWord_Op<string mnemonic, list<Trait> traits = []> :
        Op<JaxiteWord_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::jaxiteword";
}

class JaxiteWord_UnaryTypeSwitchOp<string mnemonic, list<Trait> traits = []>
  : JaxiteWord_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext
  );
  let results = (outs LWECiphertext:$output);
}

class JaxiteWord_UnaryOp<string mnemonic, list<Trait> traits = []>
  : JaxiteWord_UnaryTypeSwitchOp<
      mnemonic, traits # [AllTypesMatch<["ciphertext", "output"]>]
    >;

class JaxiteWord_BinaryOp<string mnemonic, list<Trait> traits = []>
  : JaxiteWord_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

// In-place binary operation: modifies lhs, no return value
class JaxiteWord_BinaryInPlaceOp<string mnemonic, list<Trait> traits = []>
  : JaxiteWord_Op<mnemonic, traits # [AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "In-place binary operation for JaxiteWord";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs
  );
}

// Context and Key Generation Ops
def GenParamsOp : JaxiteWord_Op<"gen_params"> {
  let summary = "Generate crypto context for CROSS";
  let description = [{
    Creates scheme parameters for the CROSS CKKS implementation.

    Scheme parameters (from CKKSContext):
    - degree: Ring polynomial degree (N)
    - numSlots: Number of slots (usually N/2)
    - scalingFactor: CKKS scaling factor (Δ)
    - qTowers: Modulus chain for ciphertext (Q)
    - pTowers: Modulus chain for key switching (P)

    TPU-specific parameters (from HEMul/HERot):
    - batch: Batch size for TPU
    - r, c: Tiling dimensions (r * c = degree)
    - dnum: Key switching decomposition parameter
    - numEvalMult: Number of evaluation multiplications
  }];
  let arguments = (ins
    // Scheme parameters
    I64Attr:$degree,
    I64Attr:$numSlots,
    F64Attr:$scalingFactor,
    DenseI64ArrayAttr:$qTowers,
    DenseI64ArrayAttr:$pTowers,
    // TPU tiling parameters
    I32Attr:$batch,
    I32Attr:$r,
    I32Attr:$c,
    I32Attr:$dnum,
    I32Attr:$numEvalMult,
    I32Attr:$compositeDegree
  );
  let results = (outs JaxiteWord_CryptoContext:$cryptoContext);
}

def GenKeyPairOp : JaxiteWord_Op<"gen_keypair"> {
  let summary = "Generate public/private key pair";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext
  );
  let results = (outs JaxiteWord_PublicKey:$publicKey,
                      JaxiteWord_PrivateKey:$secretKey);
}

def GenMulKeyOp : JaxiteWord_Op<"gen_mulkey"> {
  let summary = "Generate evaluation key for relinearization";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    JaxiteWord_PrivateKey:$secretKey
  );
  let results = (outs JaxiteWord_EvalKey:$evalKey);
}

def GenRotKeyOp : JaxiteWord_Op<"gen_rot_key"> {
  let summary = "Generate rotation keys";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    JaxiteWord_PrivateKey:$secretKey,
    DenseI64ArrayAttr:$indices
  );
  let results = (outs JaxiteWord_EvalKey:$rotKeys);
}

def ProgramInitializationOp : JaxiteWord_Op<"program_initialization"> {
  let summary = "Initialize CROSS program context with HEMul/HERot accessors";
  let description = [{
    Calls ctx.program_initialization() which pre-builds level-indexed
    HEMul and HERot accessors and loads all keys onto the context.
    Replaces separate GenMulKeyOp + GenRotKeyOp for the new CROSS API.
  }];
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    JaxiteWord_PrivateKey:$secretKey,
    I64Attr:$totalHemulLevels,
    DenseI64ArrayAttr:$totalRotationIndices,
    I32Attr:$dnum,
    I32Attr:$r,
    I32Attr:$c,
    I32Attr:$batch
  );
}

def EncodeOp : JaxiteWord_Op<"encode", [Pure]> {
  let summary = "Encode a real tensor into a plaintext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    AnyTypeOf<[AnyShaped, AnyFloat]>:$input
  );
  let results = (outs LWEPlaintext:$plaintext);
}

def DecodeOp : JaxiteWord_Op<"decode", [Pure]> {
  let summary = "Decode a plaintext back to real values";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWEPlaintext:$plaintext
  );
  let results = (outs AnyTypeOf<[AnyShaped, AnyFloat]>:$output);
}

def EncryptOp : JaxiteWord_Op<"encrypt", [Pure]> {
  let summary = "Encrypt a plaintext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWEPlaintext:$plaintext,
    JaxiteWord_PublicKey:$publicKey
  );
  let results = (outs LWECiphertext:$ciphertext);
}

def DecryptOp : JaxiteWord_Op<"decrypt", [Pure]> {
  let summary = "Decrypt a ciphertext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    JaxiteWord_PrivateKey:$secretKey
  );
  let results = (outs LWEPlaintext:$plaintext);
}

def AddOp : JaxiteWord_BinaryOp<"add", [Commutative]> {
  let summary = "Homomorphic addition of two ciphertexts";
}

def MulOp : JaxiteWord_Op<"mul", [Pure]> {
  let summary = "Homomorphic multiplication of two ciphertexts with relinearization";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs,
    JaxiteWord_EvalKey:$evalKey
  );
  let results = (outs LWECiphertext:$output);
}

def MulNoRelinOp : JaxiteWord_Op<"mul_no_relin", [Pure]> {
  let summary = "Homomorphic multiplication without relinearization";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

def RelinOp : JaxiteWord_Op<"relin", [Pure]> {
  let summary = "Relinearize a ciphertext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    JaxiteWord_EvalKey:$evalKey
  );
  let results = (outs LWECiphertext:$output);
}

def ModReduceOp : JaxiteWord_UnaryTypeSwitchOp<"mod_reduce"> {
  let summary = "Rescale/modulus reduce a ciphertext (CKKS)";
}

def RotOp : JaxiteWord_Op<"rot", [Pure, AllTypesMatch<["ciphertext", "output"]>]> {
  let summary = "Rotate slots in a ciphertext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    JaxiteWord_EvalKey:$evalKey,
    I64Attr:$index
  );
  let results = (outs LWECiphertext:$output);
}

def SubOp : JaxiteWord_BinaryOp<"sub", [SameOperandsAndResultRings]> {
  let summary = "Homomorphic subtraction of two ciphertexts";
}

def NegateOp : JaxiteWord_UnaryOp<"negate"> {
  let summary = "Negate a ciphertext";
}

def SquareOp : JaxiteWord_UnaryOp<"square"> {
  let summary = "Square a ciphertext";
}

def AddInPlaceOp : JaxiteWord_BinaryInPlaceOp<"add_inplace"> {
  let summary = "In-place homomorphic addition";
}

def SubInPlaceOp : JaxiteWord_BinaryInPlaceOp<"sub_inplace"> {
  let summary = "In-place homomorphic subtraction";
}

def AddPlainOp : JaxiteWord_Op<"add_plain", [Pure, AllCiphertextTypesMatch]> {
  let summary = "Add plaintext to ciphertext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWEPlaintextOrCiphertext:$lhs,
    LWEPlaintextOrCiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

def SubPlainOp : JaxiteWord_Op<"sub_plain", [Pure, AllCiphertextTypesMatch]> {
  let summary = "Subtract plaintext from ciphertext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWEPlaintextOrCiphertext:$lhs,
    LWEPlaintextOrCiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

def MulPlainOp : JaxiteWord_Op<"mul_plain", [Pure]> {
  let summary = "Multiply ciphertext with plaintext";
  let arguments = (ins
    JaxiteWord_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    LWEPlaintext:$plaintext
  );
  let results = (outs LWECiphertext:$output);
}

#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_TD_
