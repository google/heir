#include "lib/Target/CompilationTarget/RegisterAllBackends.h"

// IWYU pragma: begin_keep
#include "lib/Target/CompilationTarget/CompilationTarget.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#include "lib/Target/Cheddar/cheddar_backend_config.cpp.inc"
#include "lib/Target/Lattigo/lattigo_backend_config.cpp.inc"
#include "lib/Target/OpenFhePke/openfhe_backend_config.cpp.inc"

void registerAllBackends() {
  registerTargetCheddar();
  registerTargetLattigo();
  registerTargetOpenFHE();
}

}  // namespace heir
}  // namespace mlir
