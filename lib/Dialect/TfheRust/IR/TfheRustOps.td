#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_

include "TfheRustDialect.td"
include "TfheRustTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TfheRust_Op<string mnemonic, list<Trait> traits = []> :
        Op<TfheRust_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::tfhe_rust";
}

/*******************
 * BOOL OPERATIONS *
 *******************/

class TfheRust_BinaryOp<string mnemonic>
  : TfheRust_Op<mnemonic, [
    Pure,
    Commutative,
    ElementwiseMappable,
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextLikeType:$lhs,
    TfheRust_CiphertextLikeType:$rhs
  );
  let results = (outs TfheRust_CiphertextLikeType:$output);
}

def TfheRust_BitAndOp : TfheRust_BinaryOp<"bitand"> { let summary = "Logical AND of two tfhe ciphertexts."; }
def TfheRust_BitOrOp : TfheRust_BinaryOp<"bitor"> { let summary = "Logical OR of two tfhe ciphertexts."; }
def TfheRust_BitXorOp : TfheRust_BinaryOp<"bitxor"> { let summary = "Logical XOR of two tfhe ciphertexts."; }

/********************
 * ARITH OPERATIONS *
 ********************/

class TfheRust_ScalarBinaryOp<string mnemonic>
  : TfheRust_Op<mnemonic, [
    Pure,
    Commutative,
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$lhs,
    AnyTypeOf<[Builtin_Integer, TfheRust_CiphertextType]>:$rhs
  );
  let results = (outs TfheRust_CiphertextType:$output);
  let summary = "Arithmetic sub of two tfhe ciphertexts.";
}

def TfheRust_AddOp : TfheRust_ScalarBinaryOp<"add"> { let summary = "Arithmetic add of two tfhe ciphertexts."; }
def TfheRust_MulOp : TfheRust_ScalarBinaryOp<"mul"> { let summary = "Arithmetic mul of two tfhe ciphertexts."; }

def TfheRust_SubOp : TfheRust_Op<"sub", [
    Pure,
    AllTypesMatch<["lhs","output"]>
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextLikeType:$lhs,
    AnyTypeOf<[Builtin_Integer, TfheRust_CiphertextType]>:$rhs
  );
  let results = (outs TfheRust_CiphertextType:$output);
  let summary = "Arithmetic sub of two tfhe ciphertexts.";
}

def TfheRust_ScalarLeftShiftOp : TfheRust_Op<"scalar_left_shift", [
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$ciphertext,
    IndexAttr:$shiftAmount
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

def TfheRust_ScalarRightShiftOp : TfheRust_Op<"scalar_right_shift", [
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$ciphertext,
    IndexAttr:$shiftAmount
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

/**********************
 * COMPARE OPERATIONS *
 **********************/

def TfheRust_EqOp : TfheRust_BinaryOp<"eq"> { let summary = "High level operation to check equality of two ciphertexts."; }
def TfheRust_NeqOp : TfheRust_BinaryOp<"neq"> { let summary = "High level operation to check inequality of two ciphertexts."; }
def TfheRust_MinOp : TfheRust_BinaryOp<"min"> { let summary = "High level operation to return minimum of two ciphertexts."; }
def TfheRust_MaxOp : TfheRust_BinaryOp<"max"> { let summary = "High level operation to return maximum of two ciphertexts."; }
def TfheRust_CmpOp : TfheRust_Op<"cmp", [
    Pure,
]> {
  let arguments = (ins
  TfheRust_ServerKey:$serverKey,
  Arith_CmpIPredicateAttr:$predicate,
  TfheRust_CiphertextType:$lhs,
  TfheRust_CiphertextType:$rhs);
  let results = (outs TfheRust_EncryptedBool:$output);
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
}];
}

 /******************
 * PBS OPERATIONS *
 ******************/

 def TfheRust_ApplyLookupTableOp : TfheRust_Op<"apply_lookup_table", [
    Pure,
    AllTypesMatch<["input", "output"]>
]> {
  let arguments = (
    ins TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$input,
    TfheRust_LookupTable:$lookupTable
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

def TfheRust_GenerateLookupTableOp : TfheRust_Op<"generate_lookup_table", [Pure]> {
  let arguments = (
    ins TfheRust_ServerKey:$serverKey,
    // The integer represents a binary-valued truth table as a bit string,
    // evaluated via `(lut >> input) & 1`
    Builtin_IntegerAttr:$truthTable
  );
  let results = (outs TfheRust_LookupTable:$lookupTable);
  let hasCanonicalizer = 1;
}


def TfheRust_SelectOp : TfheRust_Op<"cmux", [
    Pure
]> {
  let arguments = (
    ins TfheRust_ServerKey:$serverKey,
  TfheRust_EncryptedBool:$select,
  TfheRust_CiphertextLikeType:$trueCtxt,
  TfheRust_CiphertextLikeType:$falseCtxt
  );
  let results = (outs TfheRust_CiphertextLikeType:$output);
  let summary = [{Multiplexer operations, the `select` ciphertext will return the `trueCtxt`
  if in contains a 1. In the other case, it will will return the `falseCtxt`.
  }];
}

 /**************************
 * MAINTENANCE OPERATIONS *
 **************************/

def TfheRust_CreateTrivialOp : TfheRust_Op<"create_trivial", [Pure]> {
  let arguments = (ins TfheRust_ServerKey:$serverKey, Optional<AnyInteger>:$value, OptionalAttr<AnyAttrOf<[Builtin_IntegerAttr, Builtin_ArrayAttr]>>:$valueAttr);
  let results = (outs TfheRust_CiphertextLikeType:$output);
  let hasCanonicalizer = 1;
}

def TfheRust_CastOp : TfheRust_Op<"cast", [
    Pure,
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$ciphertext
  );
  let results = (outs TfheRust_CiphertextType:$output);
}


#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
