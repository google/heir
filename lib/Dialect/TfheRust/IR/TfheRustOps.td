#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_

include "TfheRustDialect.td"
include "TfheRustTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TfheRust_Op<string mnemonic, list<Trait> traits = []> :
        Op<TfheRust_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::tfhe_rust";
}

class TfheRust_BinaryOp<string mnemonic>
  : TfheRust_Op<mnemonic, [
    Pure,
    AllTypesMatch<["lhs", "rhs", "output"]>
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$lhs,
    TfheRust_CiphertextType:$rhs
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

def CreateTrivialOp : TfheRust_Op<"create_trivial", [Pure]> {
  let arguments = (ins TfheRust_ServerKey:$serverKey, AnyInteger:$value);
  let results = (outs TfheRust_CiphertextType:$output);
  let hasCanonicalizer = 1;
}

def TfheRust_BitAndOp : TfheRust_BinaryOp<"bitand"> { let summary = "Logical AND of two tfhe ciphertexts."; }
def TfheRust_AddOp : TfheRust_BinaryOp<"add"> { let summary = "Arithmetic add of two tfhe ciphertexts."; }
def TfheRust_SubOp : TfheRust_BinaryOp<"sub"> { let summary = "Arithmetic sub of two tfhe ciphertexts."; }


def ScalarLeftShiftOp : TfheRust_Op<"scalar_left_shift", [
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$ciphertext,
    AnyI8:$shiftAmount
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

def ApplyLookupTableOp : TfheRust_Op<"apply_lookup_table", [
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

def GenerateLookupTableOp : TfheRust_Op<"generate_lookup_table", [Pure]> {
  let arguments = (
    ins TfheRust_ServerKey:$serverKey,
    // TODO(#246): Generalize to support integer-valued lookup tables; for now
    // the integer represents a binary-valued truth table as a bit string,
    // evaluated via `(lut >> input) & 1`
    Builtin_IntegerAttr:$truthTable
  );
  let results = (outs TfheRust_LookupTable:$lookupTable);
  let hasCanonicalizer = 1;
}

#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
