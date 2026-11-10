#ifndef LIB_TOOLS_HEIROPTREGISTRATION_H_
#define LIB_TOOLS_HEIROPTREGISTRATION_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Inserts all dialects, dialect extensions, and external interface models
// that heir-opt supports into the given registry.
void registerAllHEIRDialects(DialectRegistry& registry);

// Registers all passes and pass pipelines that heir-opt supports with the
// global pass registry, except the ones that depend on Yosys and ABC (see
// HeirYosysRegistration.h). Must be called at most once per process.
void registerAllHEIRPasses();

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TOOLS_HEIROPTREGISTRATION_H_
