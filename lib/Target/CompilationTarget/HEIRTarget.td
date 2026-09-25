#ifndef LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
#define LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int has_kernel_chebyshev = 0;
  int has_kernel_linear_transform = 0;
  // Whether ciphertext-plaintext operations require equal modulus-chain levels.
  int requires_matching_ciphertext_plaintext_levels = 0;
  int can_emit_adjust_scale = 1;
}

#endif // LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
