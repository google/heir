#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_

include "OpenfheDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Openfhe_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Openfhe_Dialect, name, traits # [OpAsmTypeInterface]> {
  let mnemonic = typeMnemonic;

  string asmName = ?;
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("}] # asmName # [{");
    }

    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # asmName # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def Openfhe_PublicKey : Openfhe_Type<"PublicKey", "public_key"> {
  let summary = "The public key required to encrypt plaintext in OpenFHE.";
  let asmName = "pk";
}

def Openfhe_PrivateKey : Openfhe_Type<"PrivateKey", "private_key"> {
  let summary = "The private key required to decrypt a ciphertext in OpenFHE.";
  let asmName = "sk";
}

def Openfhe_PublicKeyOrPrivateKey : AnyTypeOf<[Openfhe_PublicKey, Openfhe_PrivateKey]>;

def Openfhe_EvalKey : Openfhe_Type<"EvalKey", "eval_key"> {
  let summary = "The evaluation key required to keyswitch/relinearize/rotate/automorphism operation in OpenFHE.";
  let asmName = "ek";
}

def Openfhe_CCParams : Openfhe_Type<"CCParams", "cc_params"> {
  let summary = "The CCParams required to create CryptoContext.";
  let asmName = "params";
}

def Openfhe_CryptoContext : Openfhe_Type<"CryptoContext", "crypto_context"> {
  let summary = "The CryptoContext required to perform homomorphic operations in OpenFHE.";
  let asmName = "cc";
}

def Openfhe_BinFHEContext : Openfhe_Type<"BinFHEContext", "binfhe_context"> {
  let summary = "The BinFHEContext required to perform homomorphic operations in OpenFHE.";
  let asmName = "ctx";
}

def Openfhe_LWEScheme : Openfhe_Type<"LWEScheme", "lwe_scheme"> {
  let summary = "A reference to the underlying LWE scheme for lower level operations in OpenFHE.";
  let asmName = "scheme";
}

def Openfhe_LUT : Openfhe_Type<"LookupTable", "lookup_table"> {
  let summary = "A lookup table mapping Z/p -> Z/p";
  let asmName = "lut";
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_TD_
