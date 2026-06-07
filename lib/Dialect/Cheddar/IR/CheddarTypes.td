#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_TD_

include "CheddarDialect.td"

include "lib/Dialect/HEIRInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"

// A base class for all types in this dialect
//
// All Cheddar types implement MemRefElementTypeInterface so they can appear as
// element types in `tensor<Nx!cheddar.*>` / `memref<Nx!cheddar.*>`, which is
// what the bufferization pipeline produces when lowering looped CKKS kernels.
class Cheddar_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cheddar_Dialect, name,
              traits # [OpAsmTypeInterface, MemRefElementTypeInterface]> {
  let mnemonic = typeMnemonic;
  let genMnemonicAlias = 1;

  string asmName = ?;
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("}] # asmName # [{");
    }
  }];
}

// Context types

def Cheddar_Context : Cheddar_Type<"Context", "context"> {
  let description = [{
    This type represents a CHEDDAR Context, the main server-side computation
    engine. Created via Context::Create(param).
  }];
  let asmName = "ctx";
}

def Cheddar_BootContext : Cheddar_Type<"BootContext", "boot_context"> {
  let description = [{
    This type represents a CHEDDAR BootContext: a Context extended with
    bootstrapping support, created via BootContext::Create(param, boot_param).
    A BootContext is-a Context in CHEDDAR, so it is accepted anywhere a Context
    is via the `Cheddar_AnyContext` constraint; only `cheddar.boot` requires it
    specifically (so the emitter can call `BootContext::Boot` with no downcast).
  }];
  let asmName = "boot_ctx";
}

// Most server-side ops run on either a plain Context or a BootContext (the
// latter derives from the former in CHEDDAR). `cheddar.boot` is the exception:
// it requires a Cheddar_BootContext.
def Cheddar_AnyContext : AnyTypeOf<[Cheddar_Context, Cheddar_BootContext]>;

def Cheddar_Parameter : Cheddar_Type<"Parameter", "parameter"> {
  let description = [{
    This type represents a CHEDDAR Parameter object, constructed from a JSON
    file or programmatically.
  }];
  let asmName = "param";
}

def Cheddar_Encoder : Cheddar_Type<"Encoder", "encoder"> {
  let description = [{
    This type represents the CHEDDAR Encoder, accessed via context->encoder_.
    Used for encoding/decoding plaintext values.
  }];
  let asmName = "encoder";
}

def Cheddar_UserInterface : Cheddar_Type<"UserInterface", "user_interface"> {
  let description = [{
    This type represents the CHEDDAR UserInterface, used for key generation,
    encryption, and decryption. Note: this is for test purposes only and is
    not security-hardened.
  }];
  let asmName = "ui";
}

// Data types

def Cheddar_Ciphertext : Cheddar_Type<"Ciphertext", "ciphertext", [SecretTypeInterface]> {
  let description = [{
    This type represents a CHEDDAR Ciphertext<word>. Move-only, lives on GPU.
  }];
  let asmName = "ct";
}

def Cheddar_Plaintext : Cheddar_Type<"Plaintext", "plaintext"> {
  let description = [{
    This type represents a CHEDDAR Plaintext<word>. Contains an NTT-applied
    encoded message.
  }];
  let asmName = "pt";
}

def Cheddar_Constant : Cheddar_Type<"Constant", "constant"> {
  let description = [{
    This type represents a CHEDDAR Constant<word>. A scalar in RNS form,
    used for efficient ciphertext-scalar operations.
  }];
  let asmName = "const";
}

// Key types

def Cheddar_EvalKey : Cheddar_Type<"EvalKey", "eval_key"> {
  let description = [{
    This type represents a single CHEDDAR EvaluationKey<word>.
  }];
  let asmName = "evk";
}

def Cheddar_EvkMap : Cheddar_Type<"EvkMap", "evk_map"> {
  let description = [{
    This type represents a CHEDDAR EvkMap<word>, which bundles all evaluation
    keys (multiplication, rotation, conjugation, etc.) into a single map.
  }];
  let asmName = "evk_map";
}

// Type aliases for op constraints
def Cheddar_CiphertextOrPlaintext : AnyTypeOf<[Cheddar_Ciphertext, Cheddar_Plaintext]>;

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_TD_
