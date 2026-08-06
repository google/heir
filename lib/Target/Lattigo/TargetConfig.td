include "lib/Target/CompilationTarget/HEIRTarget.td"

def Lattigo : CompilationTarget<"lattigo"> {
  let bootstrapLevelsConsumed = 16;
  let has_kernel_chebyshev = 1;
}
