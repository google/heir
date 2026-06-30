#ifndef LIB_DIALECT_POULPY_IR_POULPYTYPES_TD_
#define LIB_DIALECT_POULPY_IR_POULPYTYPES_TD_

include "PoulpyDialect.td"
include "PoulpyEnums.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "lib/Dialect/HEIRInterfaces.td"

class Poulpy_Type<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<Poulpy_Dialect, name, traits> {
      let mnemonic = typeMnemonic;
  }

// maps to Module<BE>
def Poulpy_ModuleType : Poulpy_Type<"Module", "module"> {
  let summary = "An evaluator context for the poulpy backend";
  let description = [{
    See https://docs.rs/poulpy-hal/latest/poulpy_hal/layouts/struct.Module.html
  }];
  let parameters = (ins
    DefaultValuedParameter<"::mlir::heir::poulpy::PoulpyBackend",
    "::mlir::heir::poulpy::PoulpyBackend::FFT64Ref">:$backend
  );
  let assemblyFormat = "`<` $backend `>`";
}

// maps to CKKSCiphertext<D, Normalized>
def Poulpy_CiphertextType : Poulpy_Type<"Ciphertext", "ciphertext",
  [MemRefElementTypeInterface, SecretTypeInterface]> {
  let summary = "CKKS ciphertext storage plus semantic precision metadata.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/layouts/ciphertext/struct.CKKSCiphertext.html
  }];
}

// maps to CKKSCiphertext<D, Unnormalized>
// pub type UnnormalizedCKKSCiphertext<D> = CKKSCiphertext<D, Unnormalized>;
def Poulpy_UnnormalizedCiphertextType : Poulpy_Type<"UnnormalizedCiphertext", "unnormalized_ciphertext",
  [MemRefElementTypeInterface, SecretTypeInterface]> {
  let summary = "A CKKS ciphertext produced by an unnormalized linear operation";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/layouts/ciphertext/type.UnnormalizedCKKSCiphertext.html
  }];
}

// maps to CKKSPlaintext<D>
def Poulpy_PlaintextType : Poulpy_Type<"Plaintext", "plaintext",
  [MemRefElementTypeInterface, PlaintextTypeInterface]> {
  let summary = "A plaintext in the ZNX (torus) domain in the poulpy backend";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/layouts/plaintext/struct.CKKSPlaintext.html
  }];
}

// maps to ScratchOwned<BE>
def Poulpy_ScratchType : Poulpy_Type<"Scratch", "scratch", [MemRefElementTypeInterface]> {
  let summary = "Owned scratch buffer for temporary workspace during polynomial operations.";
  let description = [{
    See https://docs.rs/poulpy-hal/latest/poulpy_hal/layouts/struct.ScratchOwned.html
  }];
}

// maps to GLWESecretPrepared<BE::OwnedBuf, BE>
def Poulpy_SecretKeyType : Poulpy_Type<"SecretKey", "secret_key", [MemRefElementTypeInterface]> {
  let summary = "A DFT-domain prepared GLWE secret key";
  let description = [{
    See https://docs.rs/poulpy-core/latest/poulpy_core/layouts/prepared/struct.GLWESecretPrepared.html
  }];
}

// maps to GLWETensorKeyPrepared<BE::OwnedBuf, BE>
def Poulpy_TensorKeyType : Poulpy_Type<"TensorKey", "tensor_key", [MemRefElementTypeInterface]> {
  let summary = "DFT-domain (prepared) variant of a GLWE tensor key.";
  let description = [{
    See https://docs.rs/poulpy-core/latest/poulpy_core/layouts/prepared/struct.GLWETensorKeyPrepared.html
  }];
}

// maps to HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>> (implements GLWEAutomorphismKeyHelper)
def Poulpy_AutomorphismKeyMapType : Poulpy_Type<"AutomorphismKeyMap", "automorphism_key_map", [MemRefElementTypeInterface]> {
  let summary = "Map of prepared automorphism keys keyed by Galois element";
  let description = [{
    See https://docs.rs/poulpy-core/latest/poulpy_core/layouts/struct.GLWEAutomorphismKey.html
  }];
}

// maps to BootstrappingContext<BE, F>
def Poulpy_BootstrappingContextType : Poulpy_Type<"BootstrappingContext", "bootstrapping_context", [MemRefElementTypeInterface]> {
  let summary = "Compiled bootstrapping context with resident DFT matrices";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/layouts/bootstrapping/struct.BootstrappingContext.html
  }];
}

// maps to BootstrappingKeysPrepared<BE::OwnedBuf, BE>
def Poulpy_BootstrappingKeysType : Poulpy_Type<"BootstrappingKeys", "bootstrapping_keys", [MemRefElementTypeInterface]> {
  let summary = "Prepared bundle of bootstrapping evaluation keys";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/layouts/bootstrapping_keys/struct.BootstrappingKeysPrepared.html
  }];
}

#endif // LIB_DIALECT_POULPY_IR_POULPYTYPES_TD_
