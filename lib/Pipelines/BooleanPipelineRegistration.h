#ifndef LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_

#include <functional>
#include <string>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#endif

namespace mlir::heir {

#ifndef HEIR_NO_YOSYS
struct MLIRToCGGIPipelineOptions : public YosysOptimizerPipelineOptions {};

using CGGIPipelineBuilder =
    std::function<void(OpPassManager &, const MLIRToCGGIPipelineOptions &)>;

CGGIPipelineBuilder mlirToCGGIPipelineBuilder(const std::string &yosysFilesPath,
                                              const std::string &abcPath);

void mlirToCGGIPipeline(OpPassManager &pm,
                        const MLIRToCGGIPipelineOptions &options,
                        const std::string &yosysFilesPath,
                        const std::string &abcPath);
#endif

struct CGGIBackendOptions : public PassPipelineOptions<CGGIBackendOptions> {
  PassOptions::Option<int> parallelism{
      *this, "parallelism",
      llvm::cl::desc(
          "batching size for parallelism. A value of -1 (default) is infinite "
          "parallelism"),
      llvm::cl::init(-1)};
};

using CGGIBackendPipelineBuilder = std::function<void(OpPassManager &)>;

using JaxiteBackendPipelineBuilder =
    std::function<void(OpPassManager &, const CGGIBackendOptions &)>;

CGGIBackendPipelineBuilder toTfheRsPipelineBuilder();

CGGIBackendPipelineBuilder toFptPipelineBuilder();

JaxiteBackendPipelineBuilder toJaxitePipelineBuilder();

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
