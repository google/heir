#ifndef LIB_DIALECT_LWE_IR_NEWLWETYPES_TD_
#define LIB_DIALECT_LWE_IR_NEWLWETYPES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/LWEAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"

// A base class for all types in this dialect
class LWE_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LWE_Dialect, name, traits # [OpAsmTypeInterface]> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  string asmName = ?;
  string aliasName = "";
  string aliasSuffix = "";
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("}] # asmName # [{");
    }

  }] # !if(!ne(aliasName, ""), [{
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # aliasName # [{";
      }] # aliasSuffix # [{
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }], "");
}

// This file defines new LWE types following
// [#785](https://github.com/google/heir/issues/785).

def NewLWESecretKey : LWE_Type<"NewLWESecretKey", "new_lwe_secret_key"> {
  let summary = "A secret key for LWE";
  let parameters = (ins
    "KeyAttr":$key,
    "::mlir::heir::polynomial::RingAttr":$ring
  );
  let asmName = "sk";
  let aliasName= "skey";
  let aliasSuffix = [{ getRing().getAliasSuffix(os); }];
}

def NewLWEPublicKey : LWE_Type<"NewLWEPublicKey", "new_lwe_public_key"> {
  let summary = "A public key for LWE";
  let parameters = (ins
    "KeyAttr":$key,
    "::mlir::heir::polynomial::RingAttr":$ring
  );
  let asmName = "pk";
  let aliasName = "pkey";
  let aliasSuffix = [{ getRing().getAliasSuffix(os); }];
}

def NewLWESecretOrPublicKey : AnyTypeOf<[NewLWESecretKey, NewLWEPublicKey]>;

def NewLWEPlaintext : LWE_Type<"NewLWEPlaintext", "new_lwe_plaintext", [MemRefElementTypeInterface]> {
  let summary = "A plaintext type";
  let parameters = (ins
    "ApplicationDataAttr":$application_data,
    "PlaintextSpaceAttr":$plaintext_space
  );
  let asmName = "pt";
  let aliasName = "pt";
}

def NewLWEPlaintextLike : TypeOrValueSemanticsContainer<NewLWEPlaintext, "new-lwe-plaintext-like">;

def NewLWECiphertext : LWE_Type<"NewLWECiphertext", "new_lwe_ciphertext", [MemRefElementTypeInterface]> {
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
  let asmName = "ct";
  let aliasName = "ct";
  let aliasSuffix = [{ getCiphertextSpace().getAliasSuffix(os); }];
}

def NewLWECiphertextLike : TypeOrValueSemanticsContainer<NewLWECiphertext, "new-lwe-ciphertext-like">;

def NewLWEPlaintextOrCiphertext : AnyTypeOf<[NewLWECiphertext, NewLWEPlaintext]>;

#endif  // LIB_DIALECT_LWE_IR_NEWLWETYPES_TD_
