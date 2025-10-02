#ifndef LIB_DIALECT_LWE_IR_LWETYPES_TD_
#define LIB_DIALECT_LWE_IR_LWETYPES_TD_

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

// This file defines LWE types following
// [#785](https://github.com/google/heir/issues/785).

def LWESecretKey : LWE_Type<"LWESecretKey", "lwe_secret_key"> {
  let summary = "A secret key for LWE";
  let parameters = (ins
    "KeyAttr":$key,
    "::mlir::heir::polynomial::RingAttr":$ring
  );
  let asmName = "sk";
  let aliasName= "skey";
  let aliasSuffix = [{ getRing().getAliasSuffix(os); }];
}

def LWEPublicKey : LWE_Type<"LWEPublicKey", "lwe_public_key"> {
  let summary = "A public key for LWE";
  let parameters = (ins
    "KeyAttr":$key,
    "::mlir::heir::polynomial::RingAttr":$ring
  );
  let asmName = "pk";
  let aliasName = "pkey";
  let aliasSuffix = [{ getRing().getAliasSuffix(os); }];
}

def LWESecretOrPublicKey : AnyTypeOf<[LWESecretKey, LWEPublicKey]>;

def LWEPlaintext : LWE_Type<"LWEPlaintext", "lwe_plaintext", [MemRefElementTypeInterface]> {
  let summary = "A plaintext type";
  let parameters = (ins
    "ApplicationDataAttr":$application_data,
    "PlaintextSpaceAttr":$plaintext_space
  );
  let asmName = "pt";
  let aliasName = "pt";
}

def LWEPlaintextLike : TypeOrValueSemanticsContainer<LWEPlaintext, "lwe-plaintext-like">;

def LWECiphertext : LWE_Type<"LWECiphertext", "lwe_ciphertext", [MemRefElementTypeInterface]> {
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


def LWEPlaintextOrCiphertext : AnyTypeOf<[LWECiphertext, LWEPlaintext]>;

def LWECiphertextLike : TypeOrValueSemanticsContainer<LWECiphertext, "lwe-ciphertext-like">;
def LWEPlaintextOrCiphertextLike : TypeOrValueSemanticsContainer<LWEPlaintextOrCiphertext, "lwe-plaintext-or-ciphertext-like">;

#endif  // LIB_DIALECT_LWE_IR_LWETYPES_TD_
