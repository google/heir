#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_TD_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_TD_

include "BGVDialect.td"
include "BGVTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def SameOperandsAndResultRings:  NativeOpTrait<"SameOperandsAndResultRings"> {
  let cppNamespace = "::mlir::heir::bgv";
}

class BGV_Op<string mnemonic, list<Trait> traits = []> :
        Op<BGV_Dialect, mnemonic, traits> {

  // See https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format
  // TODO(#106): Simplify this format by adding type constraints.
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::bgv";
}

// TODO(#100): Add plaintext-ciphertext operations.

def BGV_AddOp : BGV_Op<"add", [Commutative, SameOperandsAndResultType]> {
  let summary = "Addition operation between ciphertexts.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def BGV_SubOp : BGV_Op<"sub", [SameOperandsAndResultType]> {
  let summary = "Subtraction operation between ciphertexts.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def BGV_MulOp : BGV_Op<"mul", [Commutative, SameOperandsAndResultRings, SameTypeOperands]> {
  let summary = "Multiplication operation between ciphertexts.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($x) `->` type($output)" ;

  let hasVerifier = 1;
}

def BGV_Rotate : BGV_Op<"rotate", [SameOperandsAndResultRings]> {
  let summary = "Rotate the coefficients of the ciphertext using a Galois automorphism.";

  let arguments = (ins
    Ciphertext:$x,
    I64Attr:$offset
  );

  let results = (outs
    Ciphertext:$output
  );

  let hasVerifier = 1;
}

def BGV_Negate : BGV_Op<"negate", [SameOperandsAndResultType]> {
  let summary = "Negate the coefficients of the ciphertext.";

  let arguments = (ins
    Ciphertext:$x
  );

  let results = (outs
    Ciphertext:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
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
    Ciphertext:$x,
    DenseI32ArrayAttr:$from_basis,
    DenseI32ArrayAttr:$to_basis
  );

  let results = (outs
    Ciphertext:$output
  );

  let hasVerifier = 1;
}

def BGV_ModulusSwitch : BGV_Op<"modulus_switch", [SameOperandsAndResultRings]> {
  // This must be validated against the BGV ring parameter.
  let summary = "Lower the modulus level of the ciphertext.";

  let arguments = (ins
    Ciphertext:$x,
    I64Attr:$from_level,
    I64Attr:$to_level
  );

  let results = (outs
    Ciphertext:$output
  );

  let hasVerifier = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_TD_
