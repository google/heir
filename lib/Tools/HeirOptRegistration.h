#ifndef LIB_TOOLS_HEIROPTREGISTRATION_H_
#define LIB_TOOLS_HEIROPTREGISTRATION_H_

#include <string>

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Inserts all dialects, dialect extensions, and external interface models
// that heir-opt supports into the given registry.
void registerAllHEIRDialects(DialectRegistry& registry);

// Registers all passes and pass pipelines that heir-opt supports with the
// global pass registry, except the Yosys-based ones (see
// registerHEIRYosysPipelines). Must be called at most once per process.
void registerAllHEIRPasses();

// Registers the passes and pipelines that shell out to Yosys/ABC
// (`yosys-optimizer`, `mlir-to-cggi`). `yosysFilesPath` is the directory
// containing the Yosys techmap files, and `abcPath` is the path to the ABC
// binary. Must be called at most once per process.
void registerHEIRYosysPipelines(const std::string& yosysFilesPath,
                                const std::string& abcPath);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TOOLS_HEIROPTREGISTRATION_H_
