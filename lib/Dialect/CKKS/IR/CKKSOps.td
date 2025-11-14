#ifndef LIB_DIALECT_CKKS_IR_CKKSOPS_TD_
#define LIB_DIALECT_CKKS_IR_CKKSOPS_TD_

include "CKKSDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "lib/Dialect/HEIRInterfaces.td"
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
    LWEPlaintextOrCiphertextLike:$lhs,
    LWEPlaintextOrCiphertextLike:$rhs
  );

  let results = (outs
    LWECiphertextLike:$output
  );
}

def CKKS_AddOp : CKKS_Op<"add", [ElementwiseMappable, Commutative, SameOperandsAndResultRings,
    SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Addition operation between ciphertexts.";

  let arguments = (ins
    LWECiphertextLike:$lhs,
    LWECiphertextLike:$rhs
  );

  let results = (outs
    LWECiphertextLike:$output
  );
}

def CKKS_AddPlainOp : CKKS_CiphertextPlaintextOp<"add_plain", [ElementwiseMappable, AllCiphertextTypesMatch,
      SameOperandsAndResultPlaintextTypes, Commutative, InferTypeOpAdaptor]> {
  let summary = "Addition operation between ciphertext-plaintext.";
  let hasCanonicalizer = 1;
}

def CKKS_SubOp : CKKS_Op<"sub", [ElementwiseMappable, SameOperandsAndResultRings,
    SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Subtraction operation between ciphertexts.";

  let arguments = (ins
    LWECiphertextLike:$lhs,
    LWECiphertextLike:$rhs
  );

  let results = (outs
    LWECiphertextLike:$output
  );
}

def CKKS_SubPlainOp : CKKS_CiphertextPlaintextOp<"sub_plain", [ElementwiseMappable, AllCiphertextTypesMatch,
      SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Subtraction operation between ciphertext-plaintext.";
}

def CKKS_MulOp : CKKS_Op<"mul", [ElementwiseMappable, Commutative, SameOperandsAndResultRings, InferTypeOpAdaptor]> {
  let summary = "Multiplication operation between ciphertexts.";

  let arguments = (ins
    LWECiphertextLike:$lhs,
    LWECiphertextLike:$rhs
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let hasVerifier = 1;
}

// MulPlain op result ciphertext type could be different from the input
def CKKS_MulPlainOp : CKKS_CiphertextPlaintextOp<"mul_plain", [ElementwiseMappable, InferTypeOpAdaptor, Commutative]> {
  let summary = "Multiplication operation between ciphertext-plaintext.";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def CKKS_RotateOp : CKKS_Op<"rotate", [ElementwiseMappable, AllTypesMatch<["input", "output"]>]> {
  let summary = "Rotate the coefficients of the ciphertext using a Galois automorphism.";

  let arguments = (ins
    LWECiphertextLike:$input,
    Builtin_IntegerAttr:$offset
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input))" ;
}

def CKKS_NegateOp : CKKS_Op<"negate", [ElementwiseMappable, SameOperandsAndResultType, Involution]> {
  let summary = "Negate the coefficients of the ciphertext.";

  let arguments = (ins
    LWECiphertextLike:$input
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CKKS_RelinearizeOp : CKKS_Op<"relinearize", [
  SameOperandsAndResultRings,
  SameOperandsAndResultPlaintextTypes,
  InferTypeOpAdaptor,
  DeclareOpInterfaceMethods<ElementwiseByOperandOpInterface>
]> {
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
    LWECiphertextLike:$input,
    DenseI32ArrayAttr:$from_basis,
    DenseI32ArrayAttr:$to_basis,
    Optional<RankedTensorOf<[LWECiphertextLike]>>:$keySwitchingKey
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let builders = [
    OpBuilder<(ins
      "Value":$input,
      CArg<"::mlir::DenseI32ArrayAttr">:$fromBasis,
      CArg<"::mlir::DenseI32ArrayAttr">:$toBasis), [{
      return build($_builder, $_state, input,
                   fromBasis, toBasis, {});
    }]>
  ];

  let hasVerifier = 1;
  // This format is a bit weird because even if the key is omitted, you still
  // have to declare the type with parentheses around the operands (e.g.,
  // `(!ct1) -> !ct`). I can't get the declarative format to work with two
  // different type signatures because... well mlir-tblgen just doesn't do
  // optional stuff well.
  let assemblyFormat = [{
    $input (`,` $keySwitchingKey^)? attr-dict `:` functional-type(operands, results)
  }];
}

def CKKS_RescaleOp : CKKS_Op<"rescale", [ElementwiseMappable]> {
  let summary = "Rescales the ciphertext, which is the CKKS version of modulus switching in BGV/BFV.";

  let arguments = (ins
    LWECiphertextLike:$input,
    Polynomial_RingAttr:$to_ring
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def CKKS_LevelReduceOp : CKKS_Op<"level_reduce", [ElementwiseMappable, SameOperandsAndResultPlaintextTypes, InferTypeOpAdaptor]> {
  let summary = "Lower the modulus level of the ciphertext via dropping RNS limbs.";

  let arguments = (ins
    LWECiphertextLike:$input,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

def CKKS_BootstrapOp : CKKS_Op<"bootstrap", [ElementwiseMappable]> {
  let summary = "Bootstrap the ciphertext to reduce noise and refresh its parameters.";

  let description = [{
    Bootstrapping is a technique used in FHE to reduce the noise in a ciphertext
    and refresh its parameters, allowing for further computations on the ciphertext.
  }];

  let arguments = (ins
    LWECiphertextLike:$input
  );

  let results = (outs
    LWECiphertextLike:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` qualified(type($output))" ;
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSOPS_TD_
