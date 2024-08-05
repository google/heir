#ifndef LIB_DIALECT_CKKS_IR_CKKSOPS_TD_
#define LIB_DIALECT_CKKS_IR_CKKSOPS_TD_

include "CKKSDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class CKKS_Op<string mnemonic, list<Trait> traits = []> :
        Op<CKKS_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::ckks";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

class CKKS_CiphertextPlaintextOp<string mnemonic, list<Trait> traits =
    [Pure, AllTypesMatch<["ciphertextInput", "output"]>,
    TypesMatchWith<"type of 'plaintextInput' matches encoding type of 'ciphertextInput'",
                    "ciphertextInput", "plaintextInput",
                    "lwe::RLWEPlaintextType::get($_ctxt, "
                      "::llvm::cast<lwe::RLWECiphertextType>($_self).getEncoding(),"
                      "::llvm::cast<lwe::RLWECiphertextType>($_self).getRlweParams().getRing(),"
                      "::llvm::cast<lwe::RLWECiphertextType>($_self).getUnderlyingType())">]> :
        CKKS_Op<mnemonic, traits> {
  let arguments = (ins
    RLWECiphertext:$ciphertextInput,
    RLWEPlaintext:$plaintextInput
  );

  let results = (outs
    RLWECiphertext:$output
  );
}

def CKKS_AddOp : CKKS_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "Addition operation between ciphertexts.";

  let arguments = (ins
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CKKS_AddPlainOp : CKKS_CiphertextPlaintextOp<"add_plain"> {
  let summary = "Addition operation between ciphertext-plaintext.";
}

def CKKS_SubOp : CKKS_Op<"sub", [SameOperandsAndResultType]> {
  let summary = "Subtraction operation between ciphertexts.";

  let arguments = (ins
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CKKS_SubPlainOp : CKKS_CiphertextPlaintextOp<"sub_plain"> {
  let summary = "Subtraction operation between ciphertext-plaintext.";
}

def CKKS_MulOp : CKKS_Op<"mul", [Pure, Commutative, SameOperandsAndResultRings, SameTypeOperands, InferTypeOpAdaptor]> {
  let summary = "Multiplication operation between ciphertexts.";

  let arguments = (ins
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
}

def CKKS_MulPlainOp : CKKS_CiphertextPlaintextOp<"mul_plain"> {
  let summary = "Multiplication operation between ciphertext-plaintext.";
}

def CKKS_RotateOp : CKKS_Op<"rotate", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "Rotate the coefficients of the ciphertext using a Galois automorphism.";

  let arguments = (ins
    RLWECiphertext:$input,
    Builtin_IntegerAttr:$offset
  );

  let results = (outs
    RLWECiphertext:$output
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
    RLWECiphertext:$input,
    AnySignlessIntegerOrIndex:$offset
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
}

def CKKS_NegateOp : CKKS_Op<"negate", [Pure, SameOperandsAndResultType, Involution]> {
  let summary = "Negate the coefficients of the ciphertext.";

  let arguments = (ins
    RLWECiphertext:$input
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CKKS_RelinearizeOp : CKKS_Op<"relinearize", [Pure, SameOperandsAndResultRings, InferTypeOpAdaptor]> {
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
    RLWECiphertext:$input,
    DenseI32ArrayAttr:$from_basis,
    DenseI32ArrayAttr:$to_basis
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def CKKS_RescaleOp : CKKS_Op<"rescale", [Pure]> {
  let summary = "Rescales the ciphertext, which is the CKKS version of modulus switching in BGV/BFV.";

  let arguments = (ins
    RLWECiphertext:$input,
    Polynomial_RingAttr:$to_ring
  );

  let results = (outs
    RLWECiphertext:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSOPS_TD_
