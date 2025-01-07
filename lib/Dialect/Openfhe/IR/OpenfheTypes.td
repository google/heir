#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_

include "OpenfheDialect.td"

include "lib/Utils/Tablegen/AsmInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Openfhe_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Openfhe_Dialect, name, traits # [
      DeclareTypeInterfaceMethods<TypeAsmInterface, ["suggestedName"]>
    ]> {
  let mnemonic = typeMnemonic;

  string nameSuggestion = ?;
  let extraClassDeclaration = "std::string suggestedName() {return \"" # nameSuggestion # "\"; }";
}

def Openfhe_PublicKey : Openfhe_Type<"PublicKey", "public_key"> {
  let summary = "The public key required to encrypt plaintext in OpenFHE.";
  let nameSuggestion = "pk";
}

def Openfhe_PrivateKey : Openfhe_Type<"PrivateKey", "private_key"> {
  let summary = "The private key required to decrypt a ciphertext in OpenFHE.";
  let nameSuggestion = "sk";
}

def Openfhe_EvalKey : Openfhe_Type<"EvalKey", "eval_key"> {
  let summary = "The evaluation key required to keyswitch/relinearize/rotate/automorphism operation in OpenFHE.";
  let nameSuggestion = "ek";
}

def Openfhe_CCParams : Openfhe_Type<"CCParams", "cc_params"> {
  let summary = "The CCParams required to create CryptoContext.";
  let nameSuggestion = "params";
}

def Openfhe_CryptoContext : Openfhe_Type<"CryptoContext", "crypto_context"> {
  let summary = "The CryptoContext required to perform homomorphic operations in OpenFHE.";
  let nameSuggestion = "cc";
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_
