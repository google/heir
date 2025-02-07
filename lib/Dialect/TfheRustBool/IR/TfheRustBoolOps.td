#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_TD_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_TD_

include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.td"
include "lib/Dialect/TfheRustBool/IR/TfheRustBoolTypes.td"
include "lib/Dialect/TfheRustBool/IR/TfheRustBoolAttributes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def HasServerKeyTrait : CPred<"$_self.hasTrait<::mlir::heir::tfhe_rust_bool::ServerKeyTrait>()">;

def AnyServerKeyType : Type<HasServerKeyTrait,
                                     "tfhe rust bool server key">;

class TfheRustBool_Op<string mnemonic, list<Trait> traits = []> :
        Op<TfheRustBool_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::tfhe_rust_bool";
}

def CreateTrivialOp : TfheRustBool_Op<"create_trivial", [Pure]> {
  let arguments = (ins AnyServerKeyType:$serverKey, I1:$value);
  let results = (outs TfheRustBool_Encrypted:$output);
}

// --- Operations for a gate-bootstrapping API of a CGGI library ---
def TfheRustBoolLike : TypeOrValueSemanticsContainer<TfheRustBool_Encrypted, "eb-like">;

class TfheRustBool_BinaryGateOp<string mnemonic>
  : TfheRustBool_Op<mnemonic, [
    Pure,
    ElementwiseMappable,
    Scalarizable,
    Commutative,
    AllTypesMatch<["lhs", "rhs", "output"]>
]> {
  let arguments = (ins AnyTypeOf<[TfheRustBool_ServerKey, TfheRustBool_PackedServerKey]>:$serverKey,
    TfheRustBoolLike:$lhs,
    TfheRustBoolLike:$rhs
  );
  let results = (outs TfheRustBoolLike:$output);
}

def TfheRustBool_AndOp : TfheRustBool_BinaryGateOp<"and"> { let summary = "Logical AND of two TFHE-rs Bool ciphertexts."; }
def TfheRustBool_NandOp : TfheRustBool_BinaryGateOp<"nand"> { let summary = "Logical NAND of two TFHE-rs Bool ciphertexts."; }
def TfheRustBool_OrOp  : TfheRustBool_BinaryGateOp<"or">  { let summary = "Logical OR of two TFHE-rs Bool ciphertexts."; }
def TfheRustBool_NorOp  : TfheRustBool_BinaryGateOp<"nor">  { let summary = "Logical NOR of two TFHE-rs Bool ciphertexts."; }
def TfheRustBool_XorOp : TfheRustBool_BinaryGateOp<"xor"> { let summary = "Logical XOR of two TFHE-rs Bool ciphertexts."; }
def TfheRustBool_XnorOp : TfheRustBool_BinaryGateOp<"xnor"> { let summary = "Logical XNOR of two TFHE-rs Bool ciphertexts."; }


def TfheRustBool_PackedOp : TfheRustBool_Op<"gates_packed", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "output"]>
  ]> {
  let arguments = (ins
    AnyTypeOf<[TfheRustBool_ServerKey, TfheRustBool_PackedServerKey]>:$serverKey,
    TfheRustBoolGates_Attr: $gates,
    TfheRustBoolLike:$lhs,
    TfheRustBoolLike:$rhs
  );

    let description = [{
     Operation to where different Boolean gates are executed pairwise between elements of two ciphertext arrays.

     For example,

     ```mlir
     %0 = tfhe_rust_bool.packed_gates %a, %b {gates = #tfhe_rust_bool.tfhe_rust_bool_gates<0 : i32, 4 : i32>} :
         (!tfhe_rust_bool.server_key,
          tensor<2x!tfhe_rust_bool.eb>,
          tensor<2x!tfhe_rust_bool.eb>) -> tensor<2x!tfhe_rust_bool.eb>
     ```
     applies an "and" gate to the first elements of %a and %b and an xor gate to the
     second elements.

     Mapping is defined in the BooleanGates.td file.
  }];

  let results = (outs TfheRustBoolLike:$output);
}

def TfheRustBool_NotOp : TfheRustBool_Op<"not", [
    Pure,
    AllTypesMatch<["input", "output"]>
]> {
  let arguments = (ins
    AnyTypeOf<[TfheRustBool_ServerKey, TfheRustBool_PackedServerKey]>: $serverKey,
    TfheRustBoolLike:$input
  );
  let results = (outs TfheRustBoolLike:$output);
}

def TfheRustBool_MuxOp : TfheRustBool_Op<"mux", [
    Pure,
    AllTypesMatch<["cnd", "lhs", "rhs", "output"]>
]> {
  let arguments = (ins
    AnyTypeOf<[TfheRustBool_ServerKey, TfheRustBool_PackedServerKey]>:$serverKey,
    TfheRustBool_Encrypted:$cnd,
    TfheRustBool_Encrypted:$lhs,
    TfheRustBool_Encrypted:$rhs
  );
  let results = (outs TfheRustBool_Encrypted:$output);
}

#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_TD_
