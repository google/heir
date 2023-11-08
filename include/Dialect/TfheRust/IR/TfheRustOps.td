#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_

include "TfheRustDialect.td"
include "TfheRustTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"

class TfheRust_Op<string mnemonic, list<Trait> traits = []> :
        Op<TfheRust_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::tfhe_rust";
}

def CreateTrivialOp : TfheRust_Op<"create_trivial"> {
  let arguments = (ins TfheRust_ServerKey:$serverKey, AnyInteger:$value);
  let results = (outs TfheRust_CiphertextType:$output);
}

def ScalarLeftShiftOp : TfheRust_Op<"scalar_left_shift"> {
  let arguments = (ins TfheRust_ServerKey:$serverKey, TfheRust_CiphertextType:$ciphertext, AnyI8:$shiftAmount);
  let results = (outs TfheRust_CiphertextType:$output);
}

def AddOp : TfheRust_Op<"add"> {
  let arguments = (ins
    TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$lhs,
    TfheRust_CiphertextType:$rhs
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

def ApplyLookupTableOp : TfheRust_Op<"apply_lookup_table"> {
  let arguments = (
    ins TfheRust_ServerKey:$serverKey,
    TfheRust_CiphertextType:$input,
    TfheRust_LookupTable:$lookupTable
  );
  let results = (outs TfheRust_CiphertextType:$output);
}

def GenerateLookupTableOp : TfheRust_Op<"generate_lookup_table"> {
  let arguments = (
    ins TfheRust_ServerKey:$serverKey,
    // TODO(https://github.com/google/heir/issues/246): Generalize to support
    // integer-valued lookup tables; for now the integer represents a
    // binary-valued truth table as a bit string, evaluated via
    // `(lut >> input) & 1`
    Builtin_IntegerAttr:$truthTable
  );
  let results = (outs TfheRust_LookupTable:$lookupTable);
}

#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
