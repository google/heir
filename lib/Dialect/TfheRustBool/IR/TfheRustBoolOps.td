#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_TD_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_TD_

include "TfheRustBoolDialect.td"
include "TfheRustBoolTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"


class TfheRustBool_Op<string mnemonic, list<Trait> traits = []> :
        Op<TfheRustBool_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::tfhe_rust_bool";
}

def CreateTrivialOp : TfheRustBool_Op<"create_trivial", [Pure]> {
  let arguments = (ins TfheRustBool_ServerKey:$serverKey, I1:$value);
  let results = (outs TfheRustBool_Encrypted:$output);
}

// --- Operations for a gate-bootstrapping API of a CGGI library ---
def TfheRustBoolLike : TypeOrContainer<TfheRustBool_Encrypted, "eb-like">;


class TfheRustBool_BinaryGateOp<string mnemonic>
  : TfheRustBool_Op<mnemonic, [
    Pure,
    AllTypesMatch<["lhs", "rhs", "output"]>
]> {
  let arguments = (ins TfheRustBool_ServerKey:$serverKey,
    TfheRustBoolLike:$lhs,
    TfheRustBoolLike:$rhs
  );
  let results = (outs TfheRustBoolLike:$output);
}

def AndOp : TfheRustBool_BinaryGateOp<"and"> { let summary = "Logical AND of two TFHE-rs Bool ciphertexts."; }
def NandOp : TfheRustBool_BinaryGateOp<"nand"> { let summary = "Logical NAND of two TFHE-rs Bool ciphertexts."; }
def OrOp  : TfheRustBool_BinaryGateOp<"or">  { let summary = "Logical OR of two TFHE-rs Bool ciphertexts."; }
def NorOp  : TfheRustBool_BinaryGateOp<"nor">  { let summary = "Logical NOR of two TFHE-rs Bool ciphertexts."; }
def XorOp : TfheRustBool_BinaryGateOp<"xor"> { let summary = "Logical XOR of two TFHE-rs Bool ciphertexts."; }
def XnorOp : TfheRustBool_BinaryGateOp<"xnor"> { let summary = "Logical XNOR of two TFHE-rs Bool ciphertexts."; }

def NotOp : TfheRustBool_Op<"not", [
    Pure,
    AllTypesMatch<["input", "output"]>
]> {
  let arguments = (ins
    TfheRustBool_ServerKey:$serverKey,
    TfheRustBoolLike:$input
  );
  let results = (outs TfheRustBoolLike:$output);
}

def MuxOp : TfheRustBool_Op<"mux", [
    Pure,
    AllTypesMatch<["cnd", "lhs", "rhs", "output"]>
]> {
  let arguments = (ins
    TfheRustBool_ServerKey:$serverKey,
    TfheRustBool_Encrypted:$cnd,
    TfheRustBool_Encrypted:$lhs,
    TfheRustBool_Encrypted:$rhs
  );
  let results = (outs TfheRustBool_Encrypted:$output);
}

#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_TD_
