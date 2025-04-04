#ifndef LIB_DIALECT_BGV_IR_BGVOPS_TD_
#define LIB_DIALECT_BGV_IR_BGVOPS_TD_

include "BGVDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

class BGV_Op<string mnemonic, list<Trait> traits = []> :
        Op<BGV_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::heir::bgv";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

class BGV_CiphertextPlaintextOp<string mnemonic, list<Trait> traits = []>
      : BGV_Op<mnemonic, traits # [IsCiphertextPlaintextOp, InferTypeOpAdaptor]> {
  let arguments = (ins
    NewLWEPlaintextOrCiphertext:$lhs,
    NewLWEPlaintextOrCiphertext:$rhs
  );

  let results = (outs
    NewLWECiphertext:$output
  );
}

def BGV_AddOp : BGV_Op<"add", [Commutative,
    SameOperandsAndResultRings, SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Addition operation between ciphertexts.";

  let arguments = (ins
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );

  let results = (outs
    NewLWECiphertext:$output
  );
}

def BGV_AddPlainOp : BGV_CiphertextPlaintextOp<"add_plain", [AllCiphertextTypesMatch,
        SameOperandsAndResultPlaintextTypes, Commutative]> {
  let summary = "Addition operation between ciphertext-plaintext.";
  let hasCanonicalizer = 1;
}

def BGV_SubOp : BGV_Op<"sub", [SameOperandsAndResultRings,
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

def BGV_SubPlainOp : BGV_CiphertextPlaintextOp<"sub_plain", [AllCiphertextTypesMatch,
        SameOperandsAndResultPlaintextTypes]> {
  let summary = "Subtraction operation between ciphertext-plaintext.";
}

def BGV_MulOp : BGV_Op<"mul", [Commutative, SameOperandsAndResultRings, InferTypeOpAdaptor]> {
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

def BGV_MulPlainOp : BGV_CiphertextPlaintextOp<"mul_plain", [InferTypeOpAdaptor, Commutative]> {
  let summary = "Multiplication operation between ciphertext-plaintext.";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BGV_RotateColumnsOp : BGV_Op<"rotate_cols", [AllTypesMatch<["input", "output"]>]> {
  let summary = "Rotate the columns of the coefficients of the ciphertext using a Galois automorphism.";

  let description = [{
    This operation rotates the columns of the coefficients of the ciphertext using a
    Galois automorphism.

    Often BGV scheme is instantiated with a ring of the form `Z_q[X]/(X^N + 1)` and
    plaintext modulus `t` where `N` is a power of 2 and `t` is a prime number. In
    this case, the plaintext slots can be viewed as a `2 x N/2` matrix where
    `N/2` is the number of columns and `2` is the number of rows.
  }];

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

def BGV_RotateRowsOp : BGV_Op<"rotate_rows", [AllTypesMatch<["input", "output"]>]> {
  let summary = "Rotate the rows of the coefficients of the ciphertext using a Galois automorphism.";

  let description = [{
    This operation rotates the rows of the coefficients of the ciphertext using a
    Galois automorphism.

    Often BGV scheme is instantiated with a ring of the form `Z_q[X]/(X^N + 1)` and
    plaintext modulus `t` where `N` is a power of 2 and `t` is a prime number. In
    this case, the plaintext slots can be viewed as a `2 x N/2` matrix where
    `N/2` is the number of columns and `2` is the number of rows.
  }];

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

def BGV_ExtractOp : BGV_Op<"extract", [SameOperandsAndResultRings]> {
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

def BGV_NegateOp : BGV_Op<"negate", [Involution, SameOperandsAndResultType]> {
  let summary = "Negate the coefficients of the ciphertext.";

  let arguments = (ins
    NewLWECiphertext:$input
  );

  let results = (outs
    NewLWECiphertext:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def BGV_RelinearizeOp : BGV_Op<"relinearize", [SameOperandsAndResultRings,
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

def BGV_ModulusSwitchOp : BGV_Op<"modulus_switch"> {
  let summary = "Lower the modulus level of the ciphertext via modulus switching.";

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

def BGV_LevelReduceOp : BGV_Op<"level_reduce", [SameOperandsAndResultPlaintextTypes]> {
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

#endif  // LIB_DIALECT_BGV_IR_BGVOPS_TD_
