include "lib/Target/CompilationTarget/HEIRTarget.td"

def Cheddar : CompilationTarget<"cheddar"> {
  // scale-snu/cheddar's bootstrap circuit uses a separate extension of the
  // compute modulus chain, so a bootstrap consumes no compute levels.
  let bootstrapLevelsConsumed = 0;

  // TODO(#3429): enable both kernels when cheddar backend gets them
  let has_kernel_linear_transform = 0;
  let has_kernel_chebyshev = 0;

  // Cheddar rejects ciphertext-plaintext operations with different prime sets.
  let requires_matching_ciphertext_plaintext_levels = 1;

  // scale-snu/cheddar exposes one canonical scale per level and has no
  // operation for a same-level scalar scale correction.
  let can_emit_adjust_scale = 0;
}
