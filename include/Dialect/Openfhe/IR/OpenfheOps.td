#ifndef INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
#define INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_

include "OpenfheDialect.td"
include "OpenfheTypes.td"

include "include/Dialect/LWE/IR/LWETypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Openfhe_Op<string mnemonic, list<Trait> traits = []> :
        Op<Openfhe_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::openfhe";
}

class Openfhe_UnaryOp<string mnemonic, list<Trait> traits = []>
  :  Openfhe_Op<mnemonic, traits # [
    Pure,
    AllTypesMatch<["ciphertext", "output"]>,
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext
  );
  let results = (outs RLWECiphertext:$output);
}

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

def EncryptOp : Openfhe_Op<"encrypt", [Pure]> {
  let arguments = (ins Openfhe_CryptoContext:$cryptoContext, RLWEPlaintext:$plaintext, Openfhe_PublicKey:$publicKey);
  let results = (outs RLWECiphertext:$output);
}

def AddOp : Openfhe_BinaryOp<"add"> { let summary = "OpenFHE add operation of two ciphertexts."; }
def SubOp : Openfhe_BinaryOp<"sub"> { let summary = "OpenFHE sub operation of two ciphertexts."; }

def MulOp : Openfhe_BinaryOp<"mul"> { let summary = "OpenFHE mul operation of two ciphertexts with relinearization."; }
def MulNoRelinOp : Openfhe_BinaryOp<"mul_no_relin"> {
  let summary = "OpenFHE mul operation of two ciphertexts without relinearization.";
}
def MulAndRelinOp : Openfhe_BinaryOp<"mul_and_relin"> {
  let summary = "OpenFHE mul operation of two ciphertexts followed by relinearization to the lowest level.";
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
def RelinOp : Openfhe_UnaryOp<"relin"> { let summary = "OpenFHE relinearize operation of a ciphertext."; }

def ModReduceOp : Openfhe_UnaryOp<"mod_reduce"> { let summary = "OpenFHE mod_reduce operation of a ciphertext. (used only for BGV/CKKS)"; }
def LevelReduceOp : Openfhe_UnaryOp<"level_reduce"> { let summary = "OpenFHE level_reduce operation of a ciphertext."; }

def RotOp : Openfhe_Op<"rot",[
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    I64:$index
  );
  let results = (outs RLWECiphertext:$output);
}

def AutomorphOp : Openfhe_Op<"automorph",[
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey,
    I32:$index
  );
  let results = (outs RLWECiphertext:$output);
}

def FindAutomorphIdxOp : Openfhe_Op<"find_automorph_idx",[
  Pure,
  AllTypesMatch<["idx", "automorph_idx"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    I32:$idx
  );
  let results = (outs I32:$automorph_idx);
}

def KeySwitchOp : Openfhe_Op<"keyswitch", [
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


#endif  // INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
