#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_

include "OpenfheDialect.td"
include "OpenfheTypes.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Openfhe_Op<string mnemonic, list<Trait> traits = []> :
        Op<Openfhe_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::openfhe";
}

class Openfhe_UnaryTypeSwitchOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [
    Pure,
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext
  );
  let results = (outs RLWECiphertext:$output);
}

class Openfhe_UnaryOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_UnaryTypeSwitchOp<
      mnemonic, traits # [AllTypesMatch<["ciphertext", "output"]>]
    >;

class Openfhe_BinaryOp<string mnemonic, list<Trait> traits = []>
  :  Openfhe_Op<mnemonic, traits # [
    Pure,
    AllTypesMatch<["lhs", "rhs", "output"]>,
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );
  let results = (outs RLWECiphertext:$output);
}

def GenParamsOp : Openfhe_Op<"gen_params"> {
  let arguments = (ins
    I64Attr:$mulDepth,
    I64Attr:$plainMod
  );
  let results = (outs Openfhe_CCParams:$params);
}

def GenContextOp : Openfhe_Op<"gen_context"> {
  let arguments = (ins
    Openfhe_CCParams:$params
  );
  let results = (outs Openfhe_CryptoContext:$context);
}

def GenMulKeyOp : Openfhe_Op<"gen_mulkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey
  );
}

def GenRotKeyOp : Openfhe_Op<"gen_rotkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey,
    DenseI64ArrayAttr:$indices
  );
}

def EncryptOp : Openfhe_Op<"encrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWEPlaintext:$plaintext,
    Openfhe_PublicKey:$publicKey)
  ;
  let results = (outs RLWECiphertext:$ciphertext);
}

def DecryptOp : Openfhe_Op<"decrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    Openfhe_PrivateKey:$privateKey)
  ;
  let results = (outs RLWEPlaintext:$plaintext);
}

def AddOp : Openfhe_BinaryOp<"add"> { let summary = "OpenFHE add operation of two ciphertexts."; }
def SubOp : Openfhe_BinaryOp<"sub"> { let summary = "OpenFHE sub operation of two ciphertexts."; }

def MulOp : Openfhe_BinaryOp<"mul"> { let summary = "OpenFHE mul operation of two ciphertexts with relinearization."; }

def MulNoRelinOp : Openfhe_Op<"mul_no_relin", [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "OpenFHE mul operation of two ciphertexts without relinearization.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );
  let results = (outs RLWECiphertext:$output);
}

def MulPlainOp : Openfhe_Op<"mul_plain",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "OpenFHE mul operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    RLWEPlaintext:$plaintext
  );
  let results = (outs RLWECiphertext:$output);
}

def MulConstOp : Openfhe_Op<"mul_const",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "OpenFHE mul operation of a ciphertext and a constant.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    I64:$constant
  );
  let results = (outs RLWECiphertext:$output);
}

def NegateOp : Openfhe_UnaryOp<"negate"> { let summary = "OpenFHE negate operation of a ciphertext."; }
def SquareOp : Openfhe_UnaryOp<"square"> { let summary = "OpenFHE square operation of a ciphertext."; }
def RelinOp : Openfhe_UnaryTypeSwitchOp<"relin"> { let summary = "OpenFHE relinearize operation of a ciphertext."; }

def ModReduceOp : Openfhe_UnaryTypeSwitchOp<"mod_reduce"> { let summary = "OpenFHE mod_reduce operation of a ciphertext. (used only for BGV/CKKS)"; }
def LevelReduceOp : Openfhe_UnaryTypeSwitchOp<"level_reduce"> { let summary = "OpenFHE level_reduce operation of a ciphertext."; }

def RotOp : Openfhe_Op<"rot",[
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    Builtin_IntegerAttr:$index
  );
  let results = (outs RLWECiphertext:$output);
}

def AutomorphOp : Openfhe_Op<"automorph", [
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs RLWECiphertext:$output);
}

def KeySwitchOp : Openfhe_Op<"key_switch", [
   Pure,
   AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs RLWECiphertext:$output);
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
