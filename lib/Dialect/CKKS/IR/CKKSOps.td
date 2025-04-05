#ifndef LIB_DIALECT_CKKS_IR_CKKSOPS_TD_
#define LIB_DIALECT_CKKS_IR_CKKSOPS_TD_

include "CKKSDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

class CKKS_Op<string mnemonic, list<Trait> traits = []> :
        Op<CKKS_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::heir::ckks";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

class CKKS_CiphertextPlaintextOp<string mnemonic, list<Trait> traits = []>
      : CKKS_Op<mnemonic, !listconcat(traits, [IsCiphertextPlaintextOp, InferTypeOpAdaptor])> {
  let arguments = (ins
    NewLWEPlaintextOrCiphertext:$lhs,
    NewLWEPlaintextOrCiphertext:$rhs
  );

  let results = (outs
    NewLWECiphertext:$output
  );
}

def CKKS_AddOp : CKKS_Op<"add", [Commutative, SameOperandsAndResultRings,
    SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Addition operation between ciphertexts.";

  let arguments = (ins
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );

  let results = (outs
    NewLWECiphertext:$output
  );
}

def CKKS_AddPlainOp : CKKS_CiphertextPlaintextOp<"add_plain", [AllCiphertextTypesMatch,
      SameOperandsAndResultPlaintextTypes, Commutative]> {
  let summary = "Addition operation between ciphertext-plaintext.";
  let hasCanonicalizer = 1;
}

def CKKS_SubOp : CKKS_Op<"sub", [SameOperandsAndResultRings,
    SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Subtraction operation between ciphertexts.";

  let arguments = (ins
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );

  let results = (outs
    NewLWECiphertext:$output
  );
}

def CKKS_SubPlainOp : CKKS_CiphertextPlaintextOp<"sub_plain", [AllCiphertextTypesMatch,
      SameOperandsAndResultPlaintextTypes]> {
  let summary = "Subtraction operation between ciphertext-plaintext.";
}

def CKKS_MulOp : CKKS_Op<"mul", [Commutative, SameOperandsAndResultRings, InferTypeOpAdaptor]> {
  let summary = "Multiplication operation between ciphertexts.";

  let arguments = (ins
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let hasVerifier = 1;
}

// MulPlain op result ciphertext type could be different from the input
def CKKS_MulPlainOp : CKKS_CiphertextPlaintextOp<"mul_plain", [InferTypeOpAdaptor, Commutative]> {
  let summary = "Multiplication operation between ciphertext-plaintext.";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def CKKS_RotateOp : CKKS_Op<"rotate", [AllTypesMatch<["input", "output"]>]> {
  let summary = "Rotate the coefficients of the ciphertext using a Galois automorphism.";

  let arguments = (ins
    NewLWECiphertext:$input,
    Builtin_IntegerAttr:$offset
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input))" ;
}

def CKKS_ExtractOp : CKKS_Op<"extract", [SameOperandsAndResultRings]> {
  let summary = "Extract the i-th element of a ciphertext.";

  let description = [{
    While this operation is costly to compute in FHE, we represent it so we can
    implement efficient lowerings and folders.

    This op can be implemented as a plaintext multiplication with a one-hot
    vector and a rotate into the zero-th index.

    An extraction op's input ciphertext type is asserted to have an `underlying_type`
    corresponding to a ranked tensor type, and this op's return type is
    inferred to have the `underlying_type` corresponding to the element type of
    that tensor type.
  }];

  let arguments = (ins
    NewLWECiphertext:$input,
    AnySignlessIntegerOrIndex:$offset
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let hasVerifier = 1;
}

def CKKS_NegateOp : CKKS_Op<"negate", [SameOperandsAndResultType, Involution]> {
  let summary = "Negate the coefficients of the ciphertext.";

  let arguments = (ins
    NewLWECiphertext:$input
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CKKS_RelinearizeOp : CKKS_Op<"relinearize", [SameOperandsAndResultRings,
    SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
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
    NewLWECiphertext:$input,
    DenseI32ArrayAttr:$from_basis,
    DenseI32ArrayAttr:$to_basis
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def CKKS_RescaleOp : CKKS_Op<"rescale"> {
  let summary = "Rescales the ciphertext, which is the CKKS version of modulus switching in BGV/BFV.";

  let arguments = (ins
    NewLWECiphertext:$input,
    Polynomial_RingAttr:$to_ring
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def CKKS_LevelReduceOp : CKKS_Op<"level_reduce", [SameOperandsAndResultPlaintextTypes]> {
  let summary = "Lower the modulus level of the ciphertext via dropping RNS limbs.";

  let arguments = (ins
    NewLWECiphertext:$input,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def CKKS_BootstrapOp : CKKS_Op<"bootstrap"> {
  let summary = "Bootstrap the ciphertext to reduce noise and refresh its parameters.";

  let description = [{
    Bootstrapping is a technique used in FHE to reduce the noise in a ciphertext
    and refresh its parameters, allowing for further computations on the ciphertext.
  }];

  let arguments = (ins
    NewLWECiphertext:$input
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSOPS_TD_
