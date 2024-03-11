#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_TD_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_TD_

include "BGVDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "include/Dialect/LWE/IR/LWETypes.td"
include "include/Dialect/Polynomial/IR/PolynomialAttributes.td"

def SameOperandsAndResultRings:  NativeOpTrait<"SameOperandsAndResultRings"> {
  let cppNamespace = "::mlir::heir::bgv";
}

class BGV_Op<string mnemonic, list<Trait> traits = []> :
        Op<BGV_Dialect, mnemonic, traits> {

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` qualified(type(operands)) `)` `->` qualified(type(results))
  }];
  let cppNamespace = "::mlir::heir::bgv";
}

// TODO(#100): Add plaintext-ciphertext operations.

def BGV_AddOp : BGV_Op<"add", [Commutative, SameOperandsAndResultType]> {
  let summary = "Addition operation between ciphertexts.";

  let arguments = (ins
    RLWECiphertext:$x,
    RLWECiphertext:$y
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($output))" ;
}

def BGV_SubOp : BGV_Op<"sub", [SameOperandsAndResultType]> {
  let summary = "Subtraction operation between ciphertexts.";

  let arguments = (ins
    RLWECiphertext:$x,
    RLWECiphertext:$y
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($output))" ;
}

def BGV_MulOp : BGV_Op<"mul", [Commutative, SameOperandsAndResultRings, SameTypeOperands]> {
  let summary = "Multiplication operation between ciphertexts.";

  let arguments = (ins
    RLWECiphertext:$x,
    RLWECiphertext:$y
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($x)) `->` qualified(type($output))" ;

  let hasVerifier = 1;
}

def BGV_Rotate : BGV_Op<"rotate", [SameOperandsAndResultRings]> {
  let summary = "Rotate the coefficients of the ciphertext using a Galois automorphism.";

  let arguments = (ins
    RLWECiphertext:$x,
    I64Attr:$offset
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
}

def BGV_Negate : BGV_Op<"negate", [SameOperandsAndResultType]> {
  let summary = "Negate the coefficients of the ciphertext.";

  let arguments = (ins
    RLWECiphertext:$x
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($output))" ;
}

def BGV_Relinearize : BGV_Op<"relinearize", [SameOperandsAndResultRings]> {
  let summary = "Relinearize the ciphertext.";

  let description = [{
    This op takes integer array attributes `from_basis` and `to_basis` that are
    used to indicate the key basis from which and to which the ciphertext is
    encrypted against. A ciphertext is canonically encrypted against key basis
    `(1, s)`. After a multiplication, its size will increase and the basis will be
    `(1, s, s^2)`. The array that represents the key basis is constructed by
    listing the powers of `s` at each position of the array. For example, `(1, s,
    s^2)` corresponds to `[0, 1, 2]`, while `(1, s^2)` corresponds to `[0, 2]`.
  }];

  let arguments = (ins
    RLWECiphertext:$x,
    DenseI32ArrayAttr:$from_basis,
    DenseI32ArrayAttr:$to_basis
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
}

def BGV_ModulusSwitch : BGV_Op<"modulus_switch"> {
  let summary = "Lower the modulus level of the ciphertext.";

  let arguments = (ins
    RLWECiphertext:$x,
    Ring_Attr:$to_ring
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_TD_
