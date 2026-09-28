#ifndef LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PrepareLinearTransforms : Pass<"prepare-linear-transforms", "ModuleOp"> {
  let summary = "Split kernel.linear_transform into prepare and apply";
  let description = [{
    Splits each `kernel.linear_transform` into a
    `kernel.prepare_linear_transform` of its diagonals and a
    `kernel.apply_linear_transform` of the resulting prepared transform, when
    the backend target supports `has_kernel_linear_transform`. Only applies to
    inputs matching `SecretLikeTypeInterface` or containers thereof, i.e.,
    LWECiphertextType or lower-level ciphertext types.

    The produced `kernel.prepare_linear_transofrm` op implements
    `PlaintextEncodeOpInterface`, so the later `split-preprocessing` pass
    hoists it into the preprocessing helper functions.

    The prepared type records the level, slot count, and BSGS split baked into
    the encoded diagonals, taken here from the consuming ciphertext's type.
    Backends currently read these off the ciphertext at runtime, where a
    mismatch silently evaluates a wrongly-scaled transform; carrying them in
    the type makes a mismatch a verifier error instead.
  }];
  let dependentDialects = ["mlir::heir::kernel::KernelDialect"];
}

#endif  // LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_
