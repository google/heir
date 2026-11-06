#ifndef LIB_TOOLS_HEIRYOSYSREGISTRATION_H_
#define LIB_TOOLS_HEIRYOSYSREGISTRATION_H_

#include <string>

namespace mlir {
namespace heir {

// Registers the passes and pipelines that shell out to Yosys and ABC
// (`yosys-optimizer`, `mlir-to-cggi`). `yosysFilesPath` is the directory
// containing the Yosys techmap files, and `abcPath` is the path to the ABC
// binary. Must be called at most once per process.
//
// Only heir-opt registers these (unless it is built with
// `--//:enable_yosys=0`); the python bindings do not.
void registerHEIRYosysPipelines(const std::string& yosysFilesPath,
                                const std::string& abcPath);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TOOLS_HEIRYOSYSREGISTRATION_H_
