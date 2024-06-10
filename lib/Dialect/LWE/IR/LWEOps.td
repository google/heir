#ifndef HEIR_LIB_DIALECT_LWE_IR_LWEOPS_TD_
#define HEIR_LIB_DIALECT_LWE_IR_LWEOPS_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/LWETypes.td"
include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

class LWE_Op<string mnemonic, list<Trait> traits = []> :
        Op<LWE_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::lwe";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

class HasEncoding<
  string encodingHolder,
  string encoding,
  string ty,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first arg's type's encoding matches the given encoding",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty # ">($" # encodingHolder # ".getType()).getEncoding(), " #
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
      "::llvm::cast<lwe::" # ty1 # ">($" # encodingHolder1 # ".getType()).getEncoding(), " #
      "::llvm::cast<lwe::" # ty2 # ">($" # encodingHolder2 # ".getType()).getEncoding())"
    >
  >;

def LWE_EncodeOp : LWE_Op<"encode", [
    Pure, HasEncoding<"output", "encoding", "LWEPlaintextType">]> {
  let summary = "Encode an integer to yield an LWE plaintext";
  let description = [{
    Encode an integer to yield an LWE plaintext.

    This op uses a an encoding attribute to encode the bits of the integer into
    an LWE plaintext value that can then be encrypted.

    Examples:

    ```
    %Y = lwe.encode %value {encoding = #enc}: i1 to !lwe.lwe_plaintext<encoding = #enc>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$plaintext,
    AnyLWEEncodingAttr:$encoding
  );

  let results = (outs LWEPlaintext:$output);
  let assemblyFormat = "$plaintext attr-dict `:` qualified(type($plaintext)) `to` qualified(type($output))";
}

def LWE_TrivialEncryptOp: LWE_Op<"trivial_encrypt", [
    Pure,
    EncodingsMatch<"input", "LWEPlaintextType", "output", "LWECiphertextType">]> {
  let summary = "Create a trivial encryption of a plaintext.";

  let arguments = (ins
    LWEPlaintext:$input,
    OptionalAttr<LWE_LWEParams>:$params
  );

  let results = (outs LWECiphertext:$output);

  let assemblyFormat = [{
    operands attr-dict `:`  qualified(type(operands)) `to` qualified(type(results))
  }];

  // Verify that the LWE params matches the output ciphertext LWE params and
  // that the encodings of the input and output match.
  let hasVerifier = 1;
}

def LWE_AddOp : LWE_Op<"add", [
    Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "Add two LWE ciphertexts";
  let arguments = (ins LWECiphertextLike:$lhs, LWECiphertextLike:$rhs);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def LWE_MulScalarOp : LWE_Op<"mul_scalar", [
    Pure, Commutative, AllTypesMatch<["ciphertext", "output"]>]> {
  let summary = "Multiply an LWE ciphertext by a scalar";
  let arguments = (ins LWECiphertextLike:$ciphertext, AnyInteger:$scalar);
  let results = (outs LWECiphertextLike:$output);
}

def LWE_RLWEEncodeOp : LWE_Op<"rlwe_encode", [Pure, HasEncoding<"output", "encoding", "RLWEPlaintextType">]> {
  let summary = "Encode an integer to yield an RLWE plaintext";
  let description = [{
    Encode an integer to yield an RLWE plaintext.

    This op uses a an encoding attribute to encode the bits of the integer into
    an RLWE plaintext value that can then be encrypted.

    Examples:

    ```
    %Y = lwe.rlwe_encode %value {encoding = #enc, ring = #ring}: i1 to !lwe.rlwe_plaintext<encoding = #enc, ring = #ring>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$input,
    AnyRLWEEncodingAttr:$encoding,
    Polynomial_RingAttr:$ring
  );

  let results = (outs RLWEPlaintext:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($output))";
}

def LWE_RLWEDecodeOp : LWE_Op<"rlwe_decode", [
    Pure, HasEncoding<"input", "encoding", "RLWEPlaintextType">]> {
  let summary = "Decode an RLWE plaintext to an underlying type";

  let arguments = (ins
    RLWEPlaintext:$input,
    AnyRLWEEncodingAttr:$encoding,
    Polynomial_RingAttr:$ring
  );

  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($output))";
}

def ReinterpretUnderlyingTypeOp : LWE_Op<"reinterpret_underlying_type", [Pure]> {
  let summary = "A placeholder cast from one ciphertext type to another";
  let description = [{
    The `cast` op is thus used to translate `underlying_type` between
    ciphertexts in particular situations , such as when lowering to an API that
    does not keep track of types for you.
  }];

  let arguments = (ins RLWECiphertext:$input);
  let results = (outs RLWECiphertext:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];

  let hasVerifier = 1;
}

#endif  // HEIR_LIB_DIALECT_LWE_IR_LWEOPS_TD_
