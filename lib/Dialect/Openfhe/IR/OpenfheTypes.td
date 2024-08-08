#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_

include "OpenfheDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Openfhe_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Openfhe_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Openfhe_PublicKey : Openfhe_Type<"PublicKey", "public_key"> {
  let summary = "The public key required to encrypt plaintext in OpenFHE.";
}

def Openfhe_PrivateKey : Openfhe_Type<"PrivateKey", "private_key"> {
  let summary = "The private key required to decrypt a ciphertext in OpenFHE.";
}

def Openfhe_EvalKey : Openfhe_Type<"EvalKey", "eval_key"> {
  let summary = "The evaluation key required to keyswitch/relinearize/rotate/automorphism operation in OpenFHE.";
}

def Openfhe_CryptoContext : Openfhe_Type<"CryptoContext", "crypto_context"> {
  let summary = "The CryptoContext required to perform homomorphic operations in OpenFHE.";
}

def Openfhe_LUT : Openfhe_Type<"LookupTable", "lookup_table"> {
  let summary = "A lookup table mapping Z/p -> Z/p";
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_
