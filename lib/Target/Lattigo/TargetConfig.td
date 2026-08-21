include "lib/Target/CompilationTarget/HEIRTarget.td"

def Lattigo : CompilationTarget<"lattigo"> {
  // Lattigo's bootstrapping circuit runs in its own parameter set, whose ring
  // carries extra primes (bootstrapping.NewParametersFromLiteral takes the
  // compute parameters as the *residual* ones), so a bootstrap returns the
  // ciphertext at the top of the compute chain and consumes none of it. A
  // level probe on tcn confirms it: Bootstrap(in=41) -> out=26 with
  // maxlevel=26. Charging levels here makes every level after a bootstrap 16
  // too low, which inflates the modulus chain and mis-schedules the rescales.
  let bootstrapLevelsConsumed = 0;
  let has_kernel_chebyshev = 1;
  let has_kernel_linear_transform = 1;
  let has_prepared_linear_transform = 1;
}
