/*******************
 * BOOL OPERATIONS *
 *******************/

class CGGI_BinaryOp<string mnemonic>
  : CGGI_Op<mnemonic, [
    Pure,
    Commutative,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins NewLWECiphertextLike:$lhs, NewLWECiphertextLike:$rhs);
  let results = (outs NewLWECiphertextLike:$output);
  // Note: error: type of result #0, named 'output', is not buildable and a buildable type cannot be inferred
  //  NewLWECiphertext is not buildable?
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CGGI_AndOp : CGGI_BinaryOp<"and"> { let summary = "Logical AND of two ciphertexts."; }
def CGGI_NandOp : CGGI_BinaryOp<"nand"> { let summary = "Logical NAND of two ciphertexts."; }
def CGGI_NorOp  : CGGI_BinaryOp<"nor">  { let summary = "Logical NOR of two ciphertexts."; }
def CGGI_OrOp  : CGGI_BinaryOp<"or">  { let summary = "Logical OR of two ciphertexts."; }
def CGGI_XorOp : CGGI_BinaryOp<"xor"> { let summary = "Logical XOR of two ciphertexts."; }
def CGGI_XNorOp : CGGI_BinaryOp<"xnor"> { let summary = "Logical XNOR of two ciphertexts."; }

def CGGI_NotOp : CGGI_Op<"not", [
    Pure,
    Involution,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins NewLWECiphertextLike:$input);
  let results = (outs NewLWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
  let summary = "Logical NOT of two ciphertexts";
}

def CGGI_PackedOp : CGGI_Op<"packed_gates",
  [Pure, AllTypesMatch<["lhs", "rhs", "output"]>]>{
  let arguments = (ins
    CGGI_CGGIBoolGates:$gates,
    NewLWECiphertextLike:$lhs,
    NewLWECiphertextLike:$rhs
  );

      let description = [{
     Operation to where different Boolean gates are executed pairwise between elements of two ciphertext arrays.

     For example,

     ```mlir
     %0 = cggi.packed_gates %a, %b {gates = #cggi.cggi_gate<"and", "xor">} : tensor<2x!lwe.lwe_ciphertext>
     ```
     applies an "and" gate to the first elements of %a and %b and an xor gate to the
     second elements.

     Mapping is defined in the BooleanGates.td file.
  }];

  let results = (outs NewLWECiphertextLike:$output);
}

/********************
 * ARITH OPERATIONS *
 ********************/

class CGGI_ScalarBinaryOp<string mnemonic>
  : CGGI_Op<mnemonic, [
    Pure,
    Commutative
]> {
  let arguments = (ins NewLWECiphertext:$lhs, AnyTypeOf<[Builtin_Integer, NewLWECiphertext]>:$rhs);
  let results = (outs NewLWECiphertext:$output);
}

def CGGI_AddOp : CGGI_ScalarBinaryOp<"add"> { let summary = "Arithmetic addition of two ciphertexts. One of the two ciphertext is allowed to be a scalar, this will result in the scalar addition to a ciphertext."; }
def CGGI_MulOp : CGGI_ScalarBinaryOp<"mul"> {
  let summary = "Arithmetic multiplication of two ciphertexts. One of the two ciphertext is allowed to be a scalar, this will result in the scalar multiplication to a ciphertext.";
  let description = [{
    While CGGI does not have a native multiplication operation,
    some backend targets provide a multiplication
    operation that is implemented via a sequence
    of other atomic CGGI ops. When lowering to
    backends that do not have this, one must lower
    to this op the appropriate CGGI ops.
  }];
}

def CGGI_SubOp : CGGI_Op<"sub", [
    Pure,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins  NewLWECiphertext:$lhs, AnyTypeOf<[Builtin_Integer,  NewLWECiphertext]>:$rhs);
  let results = (outs  NewLWECiphertext:$output);
  let summary = "Subtraction of two ciphertexts.";
}

def CGGI_ScalarShiftRightOp : CGGI_Op<"sshr", [
    Pure,
    AllTypesMatch<["lhs", "output"]>,
]> {
  let arguments = (ins NewLWECiphertextLike:$lhs, IndexAttr:$shiftAmount);
  let results = (outs NewLWECiphertextLike:$output);
  let summary = "Arithmetic shift to the right of a ciphertext by an integer. Note this operations to mirror the TFHE-rs implmementation.";
}

def CGGI_ScalarShiftLeftOp : CGGI_Op<"sshl", [
    Pure,
    AllTypesMatch<["lhs", "output"]>,
]> {
  let arguments = (ins NewLWECiphertextLike:$lhs, IndexAttr:$shiftAmount);
  let results = (outs NewLWECiphertextLike:$output);
  let summary = "Arithmetic shift to left of a ciphertext by an integer. Note this operations to mirror the TFHE-rs implmementation.";
}

/**********************
 * COMPARE OPERATIONS *
 **********************/

def CGGI_EqOp : CGGI_ScalarBinaryOp<"eq"> { let summary = "High level operation to check equality of two ciphertexts. Note this operations to mirror the TFHE-rs implmementation."; }
def CGGI_NeqOp : CGGI_ScalarBinaryOp<"neq"> { let summary = "High level operation to check inequality of two ciphertexts. Note this operations to mirror the TFHE-rs implmementation."; }
def CGGI_MinOp : CGGI_ScalarBinaryOp<"min"> { let summary = "High level operation to return minimum of two ciphertexts. Note this operations to mirror the TFHE-rs implmementation."; }
def CGGI_MaxOp : CGGI_ScalarBinaryOp<"max"> { let summary = "High level operation to return maximum of two ciphertexts. Note this operations to mirror the TFHE-rs implmementation."; }
def CGGI_CmpOp : CGGI_Op<"cmp", [
    Pure,
]> {
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate, NewLWECiphertextLike:$lhs, AnyTypeOf<[Builtin_Integer,  NewLWECiphertext]>:$rhs);
  let results = (outs NewLWECiphertextLike:$output);
  let summary = [{
  High level operation to check the relation of two ciphertexts.
  - equal (mnemonic: "eq"; integer value: 0)
  - not equal (mnemonic: "ne"; integer value: 1)
  - signed less than (mnemonic: "slt"; integer value: 2)
  - signed less than or equal (mnemonic: "sle"; integer value: 3)
  - signed greater than (mnemonic: "sgt"; integer value: 4)
  - signed greater than or equal (mnemonic: "sge"; integer value: 5)
  - unsigned less than (mnemonic: "ult"; integer value: 6)
  - unsigned less than or equal (mnemonic: "ule"; integer value: 7)
  - unsigned greater than (mnemonic: "ugt"; integer value: 8)
  - unsigned greater than or equal (mnemonic: "uge"; integer value: 9)

  Note: https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop
  Note this operations to indirect mirror the TFHE-rs implmementation.
}];
}
