#ifndef LIB_DIALECT_LWE_IR_LWEOPS_TD_
#define LIB_DIALECT_LWE_IR_LWEOPS_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWEAttributes.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"


class HasEncoding<
  string encodingHolder,
  string encoding,
  string ty,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first arg's type's encoding matches the given encoding",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty # ">($" # encodingHolder # ".getType()).getPlaintextSpace().getEncoding(), " #
      "$" # encoding # ")"
    >
  >;

class EncodingsMatch<
  string encodingHolder1,
  string ty1,
  string encodingHolder2,
  string ty2,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first arg's type's encoding matches the given encoding",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty1 # ">($" # encodingHolder1 # ".getType()).getPlaintextSpace().getEncoding(), " #
      "::llvm::cast<lwe::" # ty2 # ">($" # encodingHolder2 # ".getType()).getPlaintextSpace().getEncoding())"
    >
  >;

class KeyAndCiphertextMatch<
  string v1, string ty1, string v2, string ty2,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first value's RLWE parameters matches the second value's",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty1 # ">($" # v1 # ".getType()).getRing(), " #
      "::llvm::cast<lwe::" # ty2 # ">($" # v2 # ".getType()).getCiphertextSpace().getRing())"
    >
  >;


// LWE Operations are always Pure by design
class LWE_Op<string mnemonic, list<Trait> traits = []> :
        Op<LWE_Dialect, mnemonic,  traits # [Pure]> {
  let cppNamespace = "::mlir::heir::lwe";
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

class LWE_BinOp<string mnemonic, list<Trait> traits = []> :
        LWE_Op<mnemonic,  traits # [ElementwiseMappable]> {
  let arguments = (ins LWECiphertextLike:$lhs, LWECiphertextLike:$rhs);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

class LWE_CiphertextPlaintextOp<string mnemonic, list<Trait> traits = []> :
        LWE_Op<mnemonic, traits #  [IsCiphertextPlaintextOp, InferTypeOpAdaptor]> {
  let arguments = (ins LWEPlaintextOrCiphertextLike:$lhs, LWEPlaintextOrCiphertextLike:$rhs);
  let results = (outs LWECiphertextLike:$output);
}

def LWE_EncodeOp : LWE_Op<"encode"> {
  let summary = "Encode an integer to yield an LWE plaintext";
  let description = [{
    Encode an integer to yield an LWE plaintext.

    This op uses an overflow attribute to indicate the overflow semantics. For
    CGGI ciphertexts, this will typically be `preserve_overflow` to indicate
    that the message bits can overflow into the carry bit space.

    The LWE plaintext ring is constructed with a plaintext bit width using `plaintext_bits` and a
    polynomial modulus of `x`.

    Examples:

    ```
    %Y = lwe.encode %value {overflow = #overflow, plaintext_bits = 3}: i1 to !lwe.lwe_plaintext<application_data = #app_data, plaintext_space = #space>
    ```
  }];

  let arguments = (ins
    SignlessIntegerOrFloatLike:$input,
    IndexAttr:$plaintext_bits,
    AnyOverflowAttr:$overflow
  );

  let results = (outs LWEPlaintext:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `to` qualified(type($output))";

  // Verify that the input type and the application data are compatible.
  // Verifies that the overflow attribute is one of preserve or no overflow.
  let hasVerifier = 1;
}

def LWE_TrivialEncryptOp: LWE_Op<"trivial_encrypt", [
    EncodingsMatch<"input", "LWEPlaintextType", "output", "LWECiphertextType">]> {
  let summary = "Create a trivial encryption of a plaintext.";

  let arguments = (ins
    LWEPlaintext:$input,
    IndexAttr:$ciphertext_bits
  );

  let results = (outs LWECiphertext:$output);

  let assemblyFormat = [{
    $input attr-dict `:`  qualified(type(operands)) `to` qualified(type(results))
  }];

  // Verify that the ciphertext modulus matches the output ciphertext type's
  // ciphertext modulus and the application data of the input and output are
  // equal..
  let hasVerifier = 1;
}

def LWE_AddOp : LWE_BinOp<"add", [SameOperandsAndResultType,Commutative]> {
  let arguments = (ins LWECiphertextLike:$lhs, LWECiphertextLike:$rhs);
  let results = (outs LWECiphertextLike:$output);
  let summary = "Add two LWE ciphertexts";
}

def LWE_RAddOp : LWE_BinOp<"radd", [SameOperandsAndResultRings, SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor, Commutative]> {
  let summary = "Add two RLWE ciphertexts";
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def LWE_RAddPlainOp : LWE_CiphertextPlaintextOp<"radd_plain", [Commutative, SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Addition between RLWE ciphertext-plaintext";
  let hasCanonicalizer = 1;
}

def LWE_RSubOp : LWE_BinOp<"rsub", [SameOperandsAndResultRings, SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Subtract two RLWE ciphertexts";
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def LWE_RSubPlainOp : LWE_CiphertextPlaintextOp<"rsub_plain", [SameOperandsAndResultPlaintextTypes]> {
  let summary = "Subtraction between RLWE ciphertext-plaintext";
}

def LWE_RMulOp : LWE_BinOp<"rmul", [SameOperandsAndResultRings, InferTypeOpAdaptor, Commutative]> {
  let summary = "Multiplies two RLWE ciphertexts";
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def LWE_RMulPlainOp : LWE_CiphertextPlaintextOp<"rmul_plain", [Commutative]> {
  let summary = "Multiplication between RLWE ciphertext-plaintext";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def LWE_RNegateOp : LWE_Op<"rnegate", [SameOperandsAndResultType, ElementwiseMappable]> {
  let summary = "Negate a RLWE ciphertexts";
  let arguments = (ins LWECiphertextLike:$input);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def LWE_MulScalarOp : LWE_Op<"mul_scalar", [ElementwiseMappable,
    AllTypesMatch<["ciphertext", "output"]>]> {
  let summary = "Multiply an LWE ciphertext by a scalar";
  let arguments = (ins LWECiphertextLike:$ciphertext, AnyInteger:$scalar);
  let results = (outs LWECiphertextLike:$output);
  let hasFolder = 1;
}

def LWE_RLWEEncodeOp : LWE_Op<"rlwe_encode", [HasEncoding<"output", "encoding", "LWEPlaintextType">]> {
  let summary = "Encode an integer to yield an RLWE plaintext";
  let description = [{
    Encode an integer to yield an RLWE plaintext.

    This op uses a an encoding attribute to encode the bits of the integer into
    an RLWE plaintext value that can then be encrypted. CKKS cleartext inputs may
    be floating points, and a scaling factor described by the encoding will be
    applied.

    Examples:

    ```
    %Y = lwe.rlwe_encode %value {encoding = #enc, ring = #ring}: i1 to !lwe.rlwe_plaintext<encoding = #enc, ring = #ring>
    ```
  }];

  let arguments = (ins
    SignlessIntegerOrFloatLike:$input,
    AnyPlaintextEncodingAttr:$encoding,
    Polynomial_RingAttr:$ring
  );

  let results = (outs LWEPlaintext:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($output))";

  // Verify that the input type and the encoding are compatible.
  let hasVerifier = true;
}

def LWE_RLWEDecodeOp : LWE_Op<"rlwe_decode", [HasEncoding<"input", "encoding", "LWEPlaintextType">]> {
  let summary = "Decode an RLWE plaintext to an underlying type";

  let arguments = (ins
    LWEPlaintext:$input,
    AnyPlaintextEncodingAttr:$encoding,
    Polynomial_RingAttr:$ring
  );

  let results = (outs SignlessIntegerOrFloatLike:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($output))";

  // Verify that the expected output type and the encoding are compatible.
  let hasVerifier = true;
}

def LWE_RLWEEncryptOp : LWE_Op<"rlwe_encrypt", [
    EncodingsMatch<"input", "LWEPlaintextType", "output", "LWECiphertextType">]> {
  let summary = "Encrypt an RLWE plaintext to a RLWE ciphertext";
  let description = [{
    Encrypt an RLWE plaintext to yield a RLWE ciphertext.
  }];

  let arguments = (ins
    LWEPlaintext:$input,
    LWESecretOrPublicKey:$key
  );
  let results = (outs LWECiphertext:$output);
  let hasVerifier = 1;
}

def LWE_RLWEDecryptOp : LWE_Op<"rlwe_decrypt", [
    EncodingsMatch<"input", "LWECiphertextType", "output", "LWEPlaintextType">,
    KeyAndCiphertextMatch<"secret_key", "LWESecretKeyType", "input", "LWECiphertextType">]> {
  let summary = "Decrypt an RLWE ciphertext to a RLWE plaintext";
  let description = [{
    Decrypt an RLWE ciphertext to yield a RLWE plaintext
  }];

  let arguments = (ins
    LWECiphertext:$input,
    LWESecretKey:$secret_key
  );
  let results = (outs LWEPlaintext:$output);
}

def ReinterpretApplicationDataOp : LWE_Op<"reinterpret_application_data", [ElementwiseMappable]> {
  let summary = "A placeholder cast from one ciphertext type to another";
  let description = [{
    The `cast` op is thus used to translate application data (e.g., `message_type`)
    between ciphertexts in particular situations, such as when the bitwidth of the
    message type changes, but this change is not observed in the plaintext space,
    or when lowering to an API that does not keep track of types.
  }];

  let arguments = (ins LWECiphertextLike:$input);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];

  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_LWE_IR_LWEOPS_TD_
