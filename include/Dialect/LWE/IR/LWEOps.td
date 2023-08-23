#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_TD_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_TD_

include "LWEDialect.td"
include "LWETypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class LWE_Op<string mnemonic, list<Trait> traits = []> :
        Op<LWE_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` qualified(type(operands)) `)` `->` qualified(type(results))
  }];
  let cppNamespace = "::mlir::heir::lwe";
}

class LWE_BinOp<string mnemonic> : LWE_Op<mnemonic, [Pure, Commutative, SameOperandsAndResultType]> {
  let arguments = (ins Ciphertext:$lhs, Ciphertext:$rhs);
  let results = (outs Ciphertext:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def LWE_AddOp : LWE_BinOp<"add"> { let summary = "Addition of two ciphertexts."; }
def LWE_SubOp : LWE_BinOp<"sub"> { let summary = "Subition of two ciphertexts."; }

def LWE_NegateOp : LWE_Op<"negate", [Pure, Involution, SameOperandsAndResultType]> {
  let summary = "Negate the coefficients of the ciphertext.";
  let arguments = (ins Ciphertext:$input);
  let results = (outs Ciphertext:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def LWE_ScaleOP : LWE_Op<"scale", [Pure]> {
  let summary = "Scale the input ciphertext by a constant.";
  let arguments = (ins Ciphertext:$input, AnyInteger:$scale_by);
  let results = (outs Ciphertext:$output);
}

def LWE_ModulusSwitchOp : LWE_Op<"modulus_switch", [Pure]> {
  let summary = "Switch the modulus of the ciphertext.";
  let description = [{
  Switch the plaintext modulus of an LWE ciphertext.

  This operation increases the noise in an LWE ciphertext relative to its new
  modulus. The output ciphertext receives a different LWEEncodingScheme
  attribute to reflect the new modulus.

  For more details, see https://jeremykun.com/2022/07/16/modulus-switching-in-lwe/
  }];
  let arguments = (ins Ciphertext:$input, IndexAttr:$from_log_modulus, IndexAttr:$to_log_modulus);
  let results = (outs Ciphertext:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
  let hasVerifier = 1;
}

def LWE_KeySwitchOp : LWE_Op<"key_switch", [Pure]> {
  let summary = "Switch the secret key of an LWE ciphertext.";
  let description = [{
  Switch the secret key of an LWE ciphertext.

  This operation increases the noise in an LWE ciphertext. The keys to switch
  between are inferred from the input and output attributes. The output may
  receive a different LWEEncodingScheme attribute to reflect the parameters of
  the encoding scheme of the output secret key.

  For more details, see https://jeremykun.com/2022/08/29/key-switching-in-lwe/
  }];

  let arguments = (ins Ciphertext:$input);
  let results = (outs Ciphertext:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def LWE_EncodeOp : LWE_Op<"encode", [Pure]> {
  let summary = "Encode the input cleartext as an LWE plaintext";
  let arguments = (ins AnyInteger:$input);
  let results = (outs Plaintext:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def LWE_DecodeOp : LWE_Op<"decode", [Pure]> {
  let summary = "Decode the input LWE plaintext as a cleartext";
  let arguments = (ins Plaintext:$input);
  let results = (outs AnyInteger:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def LWE_EncryptOp : LWE_Op<"encrypt", [Pure]> {
  let summary = "Encrypt an LWE plaintext";
  let description = [{
  Encrypt an LWE plaintext.

  The secret key to encrypt with is inferred from the output type.
  }];

  let arguments = (ins Plaintext:$input);
  let results = (outs Ciphertext:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def LWE_DecryptOp : LWE_Op<"decrypt", [Pure]> {
  let summary = "Decrypt an LWE ciphertext";
  let arguments = (ins Ciphertext:$input);
  let results = (outs Plaintext:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_TD_
