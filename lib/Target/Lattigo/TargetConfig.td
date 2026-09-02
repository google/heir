include "lib/Target/CompilationTarget/HEIRTarget.td"

def Lattigo : CompilationTarget<"lattigo"> {
  // Lattigo's bootstrapping circuit runs in its own part of the parameter set,
  // whose ring carries extra primes, so a bootstrap returns the ciphertext at
  // the top of the standard compute chain and consumes none of it.
  let bootstrapLevelsConsumed = 0;
  let has_kernel_chebyshev = 1;
}
