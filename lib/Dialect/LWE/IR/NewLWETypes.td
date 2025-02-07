#ifndef LIB_DIALECT_LWE_IR_NEWLWETYPES_TD_
#define LIB_DIALECT_LWE_IR_NEWLWETYPES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/NewLWEAttributes.td"
include "lib/Utils/Tablegen/AsmInterfaces.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class LWE_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LWE_Dialect, name, traits # [
      DeclareTypeInterfaceMethods<TypeAsmInterface, ["suggestedName"]>
    ]> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  string nameSuggestion = ?;
  let extraClassDeclaration = "std::string suggestedName() {return \"" # nameSuggestion # "\"; }";
}

// This file defines new LWE types following
// [#785](https://github.com/google/heir/issues/785).

def NewLWESecretKey : LWE_Type<"NewLWESecretKey", "new_lwe_secret_key"> {
  let summary = "A secret key for LWE";
  let parameters = (ins
    "KeyAttr":$key,
    "::mlir::heir::polynomial::RingAttr":$ring
  );
  let nameSuggestion = "sk";
}

def NewLWEPublicKey : LWE_Type<"NewLWEPublicKey", "new_lwe_public_key"> {
  let summary = "A public key for LWE";
  let parameters = (ins
    "KeyAttr":$key,
    "::mlir::heir::polynomial::RingAttr":$ring
  );
  let nameSuggestion = "pk";
}

def NewLWESecretOrPublicKey : AnyTypeOf<[NewLWESecretKey, NewLWEPublicKey]>;

def NewLWEPlaintext : LWE_Type<"NewLWEPlaintext", "new_lwe_plaintext"> {
  let summary = "A plaintext type";
  let parameters = (ins
    "ApplicationDataAttr":$application_data,
    "PlaintextSpaceAttr":$plaintext_space
  );
  let nameSuggestion = "pt";
}

def NewLWEPlaintextLike : TypeOrValueSemanticsContainer<NewLWEPlaintext, "new-lwe-plaintext-like">;

def NewLWECiphertext : LWE_Type<"NewLWECiphertext", "new_lwe_ciphertext"> {
  let summary = "A ciphertext type";

  let description = [{
    An LWE ciphertext will always contain the application data, plaintext space,
    ciphertext space, and key information.

    A modulus chain is optionally specified for parameter choices in RLWE
    schemes that use more than one of modulus. When no modulus chain is
    specified, the ciphertext modulus is always the ciphertext ring's
    coefficient modulus.
  }];

  let parameters = (ins
    "ApplicationDataAttr":$application_data,
    "PlaintextSpaceAttr":$plaintext_space,
    "CiphertextSpaceAttr":$ciphertext_space,
    "KeyAttr":$key,
    OptionalParameter<"ModulusChainAttr">:$modulus_chain
  );

  let genVerifyDecl = 1;
  let nameSuggestion = "ct";
}

def NewLWECiphertextLike : TypeOrValueSemanticsContainer<NewLWECiphertext, "new-lwe-ciphertext-like">;

#endif  // LIB_DIALECT_LWE_IR_NEWLWETYPES_TD_
