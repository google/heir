include "lib/Target/CompilationTarget/HEIRTarget.td"

def Lattigo : CompilationTarget<"lattigo"> {
  let bootstrapLevelsConsumed = 16;
  let has_kernel_chebyshev = 1;
  let has_kernel_linear_transform = 1;
}
