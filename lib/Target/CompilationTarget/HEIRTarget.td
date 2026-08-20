#ifndef LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
#define LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int has_kernel_chebyshev = 0;
  int has_kernel_linear_transform = 0;
  // Whether the backend can prepare a linear transform once (encode its
  // diagonals into plaintexts) and apply it to many ciphertexts, via
  // kernel.prepare_linear_transform / kernel.apply_linear_transform.
  int has_prepared_linear_transform = 0;
}

#endif // LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
