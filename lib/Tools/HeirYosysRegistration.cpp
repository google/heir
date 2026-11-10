#include "lib/Tools/HeirYosysRegistration.h"

#include <string>

#include "lib/Pipelines/YosysPipelineRegistration.h"
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {

void registerHEIRYosysPipelines(const std::string& yosysFilesPath,
                                const std::string& abcPath) {
  registerYosysOptimizerPipeline(yosysFilesPath, abcPath);

  PassPipelineRegistration<MLIRToCGGIPipelineOptions>(
      "mlir-to-cggi",
      "Convert a func using standard MLIR dialects to FHE using "
      "CGGI.",
      mlirToCGGIPipelineBuilder(yosysFilesPath, abcPath));
}

}  // namespace heir
}  // namespace mlir
