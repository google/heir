#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_TD_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_TD_

include "include/Dialect/LWE/IR/LWEDialect.td"
include "include/Dialect/LWE/IR/LWETypes.td"
include "include/Dialect/Polynomial/IR/PolynomialAttributes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

class LWE_Op<string mnemonic, list<Trait> traits = []> :
        Op<LWE_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::lwe";
}

def LWE_EncodeOp : LWE_Op<"encode", [Pure]> {
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

  // Verify that the encoding parameter matches the output plaintext attribute.
  let hasVerifier = 1;
}

def LWE_TrivialEncryptOp: LWE_Op<"trivial_encrypt", [Pure]> {
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

def LWE_RLWEEncodeOp : LWE_Op<"rlwe_encode", [Pure]> {
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
    SignlessIntegerLike:$plaintext,
    AnyRLWEEncodingAttr:$encoding,
    Ring_Attr:$ring
  );

  let results = (outs RLWEPlaintext:$output);
  let assemblyFormat = "$plaintext attr-dict `:` qualified(type($plaintext)) `to` qualified(type($output))";

  // Verify that the encoding and ring parameter matches the output plaintext attribute.
  let hasVerifier = 1;
}


#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_TD_
