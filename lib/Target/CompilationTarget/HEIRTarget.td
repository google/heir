#ifndef LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
#define LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int hasKernelChebyshev = 0;
  int hasKernelLinearTransform = 0;
  // Whether ciphertext-plaintext operations require equal modulus-chain levels.
  int requiresMatchingCiphertextPlaintextLevels = 0;
  int supportsAdjustScale = 1;
}

#endif // LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
