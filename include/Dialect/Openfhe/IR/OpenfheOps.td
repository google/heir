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

def EncryptOp : Openfhe_Op<"encrypt", [Pure]> {
  let arguments = (ins Openfhe_CryptoContext:$cryptoContext, RLWEPlaintext:$plaintext, Openfhe_PublicKey:$publicKey);
  let results = (outs RLWECiphertext:$output);
}

def NegateOp : Openfhe_Op<"negate", [
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins Openfhe_CryptoContext:$cryptoContext, RLWECiphertext:$ciphertext);
  let results = (outs RLWECiphertext:$output);
}

def AddOp : Openfhe_Op<"add",[
    Pure,
    AllTypesMatch<["lhs", "rhs", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );
  let results = (outs RLWECiphertext:$output);
}

def SubOp : Openfhe_Op<"sub",[
    Pure,
    AllTypesMatch<["lhs", "rhs", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );
  let results = (outs RLWECiphertext:$output);
}

def MulOp : Openfhe_Op<"mul",[
    Pure,
    AllTypesMatch<["lhs", "rhs"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$lhs,
    RLWECiphertext:$rhs
  );
  let results = (outs RLWECiphertext:$output);
}
def MulPlainOp : Openfhe_Op<"mulplain",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    RLWEPlaintext:$plaintext
  );
  let results = (outs RLWECiphertext:$output);
}

def MulConstOp : Openfhe_Op<"mulconst",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    I64:$constant
  );
  let results = (outs RLWECiphertext:$output);
}

def RotOp : Openfhe_Op<"rot",[
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RLWECiphertext:$ciphertext,
    I32:$rot
  );
  let results = (outs RLWECiphertext:$output);
}
#endif  // INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
