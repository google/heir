#ifndef LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PrepareLinearTransforms : Pass<"prepare-linear-transforms", "ModuleOp"> {
  let summary = "Split kernel.linear_transform into prepare and apply";
  let description = [{
    Splits each `kernel.linear_transform` into a
    `kernel.prepare_linear_transform` of its diagonals and a
    `kernel.apply_linear_transform` of the resulting prepared transform, when
    the module's target declares `has_prepared_linear_transform`.

    Preparing a transform (row gathers, float conversion, plaintext encoding)
    is pure cleartext work that today runs inside the hot function on every
    inference; only the application needs the ciphertext. The prepare op
    implements PlaintextEncodeOpInterface, so a later split-preprocessing pass
    hoists it into the `__preprocessing` helper with no backend-specific
    machinery, and an inference pays only for the application.

    The prepared type records the level, slot count, and BSGS split baked into
    the encoded diagonals, taken here from the consuming ciphertext's type.
    Backends currently read these off the ciphertext at runtime, where a
    mismatch silently evaluates a wrongly-scaled transform; carrying them in
    the type makes a mismatch a verifier error instead.

    The pass runs after the scheme lowering has materialized ciphertext types
    (which carry the modulus chain the level is read from) and must run before
    split-preprocessing. A `kernel.linear_transform` whose input type does not
    determine a level is left untouched, as is every op when the target does
    not declare support, so `kernel.linear_transform` remains valid sugar for
    backends that lower it directly.
  }];
  let dependentDialects = ["mlir::heir::kernel::KernelDialect"];
}

#endif  // LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_
