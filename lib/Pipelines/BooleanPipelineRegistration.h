#ifndef LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_

#include <functional>
#include <string>

#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

namespace mlir::heir {

enum DataType { Bool, Integer };

// Add all Yosys optimizer pipeline options.
struct MLIRToCGGIPipelineOptions : public YosysOptimizerPipelineOptions {
  PassOptions::Option<bool> debug{
      *this, "debug",
      llvm::cl::desc("Insert debug ports after every secret operation."),
      llvm::cl::init(false)};
  PassOptions::Option<enum DataType> dataType{
      *this, "data-type",
      llvm::cl::desc("Data type to use for arithmetization."),
      llvm::cl::init(Bool),
      llvm::cl::values(
          clEnumVal(Bool, "booleanize with Yosys"),
          clEnumVal(Integer, "decompose operations into 32 bit data types"))};
};

using CGGIPipelineBuilder =
    std::function<void(OpPassManager&, const MLIRToCGGIPipelineOptions&)>;

CGGIPipelineBuilder mlirToCGGIPipelineBuilder(const std::string& yosysFilesPath,
                                              const std::string& abcPath);

void mlirToCGGIPipeline(OpPassManager& pm,
                        const MLIRToCGGIPipelineOptions& options,
                        const std::string& yosysFilesPath,
                        const std::string& abcPath);

struct CGGIBackendOptions : public PassPipelineOptions<CGGIBackendOptions> {
  PassOptions::Option<int> parallelism{
      *this, "parallelism",
      llvm::cl::desc(
          "batching size for parallelism. A value of -1 (default) is infinite "
          "parallelism"),
      llvm::cl::init(-1)};
};

using CGGIBackendPipelineBuilder = std::function<void(OpPassManager&)>;

using JaxiteBackendPipelineBuilder =
    std::function<void(OpPassManager&, const CGGIBackendOptions&)>;

CGGIBackendPipelineBuilder toTfheRsPipelineBuilder();

CGGIBackendPipelineBuilder toFptPipelineBuilder();

CGGIBackendPipelineBuilder toCGGICornamiPipelineBuilder();

JaxiteBackendPipelineBuilder toJaxitePipelineBuilder();

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
