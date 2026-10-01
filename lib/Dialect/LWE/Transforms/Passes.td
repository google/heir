#ifndef LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"


def AddDebugPort : Pass<"lwe-add-debug-port"> {
  let summary = "Add debug port to (R)LWE encrypted functions";
  let description = [{
  This pass adds debug ports to the specified function in the IR. The debug ports
  are prefixed with "__heir_debug" and are invoked after each homomorphic operation in the
  function. The debug ports are declarations and user should provide functions with
  the same name in their code.

  If the option `insert-debug-after-every-op` is set to true, it will insert a `debug.validate`
  op after every homomorphic operation.

  Regardless of the `insert-debug-after-every-op` option, this pass will lower all
  `debug.validate` ops it encounters to function calls.

  For example, if the function is called "foo", the secret key is added to its
  arguments, and the debug port is called after each homomorphic operation:
  ```mlir
  // declaration of external debug function
  func.func private @__heir_debug(%sk : !sk, %ct : !ct)

  // secret key added as function arg
  func.func @foo(%sk : !sk, ...) {
    %ct = lwe.radd ...
    // invoke external debug function
    __heir_debug(%sk, %ct)
    %ct1 = lwe.rmul ...
    __heir_debug(%sk, %ct1)
    ...
  }
  ```
  }];
  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::debug::DebugDialect"
  ];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
    Option<"minSlotCount", "min-slot-count", "int",
           /*default=*/"1", "Minimum number of slots to use for packing messages in ciphertexts.">,
    Option<"insertDebugAfterEveryOp", "insert-debug-after-every-op", "bool",
           /*default=*/"false", "Whether to add debug ports after every op">
  ];
}

def ImplementTrivialEncryptionAsAddition : Pass<"implement-trivial-encryption-as-addition"> {
  let summary = "Implement `lwe.trivial_encrypt` as ciphertext-plaintext addition";
  let description = [{
    This pass implements trivial encryption as a plaintext-ciphertext addition
    using an extra user input corresponding to an encryption of zero. This pass
    adds an additional function argument for the encryption of zero, and an
    additional client helper function to create it; this extra value must be
    additionally plumbed from the client to the compiled function.

    (* example filepath=tests/Dialect/LWE/Transforms/implement_trivial_encryption_as_addition.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
    "mlir::func::FuncDialect",
  ];

  let options = [];
}

def AnnotatePlaintextLevel : Pass<"lwe-annotate-plaintext-level", "mlir::ModuleOp"> {
  let summary = "Annotate `lwe.rlwe_encode` ops with the level they are used at";
  let description = [{
    Sets the `level` attribute of each `lwe.rlwe_encode` op to the level its
    plaintext is used at, found by walking forward from the encode op through
    plaintext-typed values.

    Backends allocate a plaintext at the top of the modulus chain unless told
    otherwise, but encoding cost is dominated by per-RNS-limb work, so a
    plaintext only ever combined with a level 3 ciphertext is cheaper to encode
    with 4 limbs than with all of them. For a ct-pt op the result is unchanged
    either way: combining a ciphertext and a plaintext yields the lower of the
    two levels, so annotating with the *highest* level among such uses is the
    smallest choice that serves them all.

    `lwe.rlwe_encrypt` is not a ct-pt op and does not follow that rule. It has
    no ciphertext operand to be clamped against, so the backend builds the
    ciphertext at the plaintext's own level and the annotation is exact rather
    than a lower bound. Annotating an encryption therefore changes what the
    client emits, not just how many limbs it encodes: where the entry level sits
    below the top of the chain, the ciphertext now enters at the level its type
    declares instead of at `MaxLevel`. A plaintext that both feeds an encryption
    and is combined with a higher-level ciphertext has no correct level, and is
    left to the fallback.

    Run this after CSE. Identical encode ops that CSE has merged are cheaper as
    one plaintext at the highest level than as several at their own levels,
    since each encode pays a fixed cost on top of the per-limb work.

    (* example filepath=tests/Dialect/LWE/Transforms/annotate_plaintext_level.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
  ];
}

def DecomposeLWEOps : Pass<"lwe-decompose"> {
  let summary = "Decomposes complex LWE ops into primitive LWE ops";
  let description = [{
  (* example filepath=tests/Dialect/LWE/Transforms/DecomposeLWEOps/keyswitch.mlir *)
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
