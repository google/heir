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

enum DataType { Bool, Integer };

#ifndef HEIR_NO_YOSYS
// If Yosys is enabled, also add all yosys optimizer pipeline options.
struct MLIRToCGGIPipelineOptions : public YosysOptimizerPipelineOptions {
  PassOptions::Option<enum DataType> dataType{
      *this, "data-type",
      llvm::cl::desc("Data type to use for arithmetization, yosys must be "
                     "enabled for Boolean."),
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

#else
struct MLIRToCGGIPipelineOptions
    : public PassPipelineOptions<MLIRToCGGIPipelineOptions> {
  PassOptions::Option<enum DataType> dataType{
      *this, "data-type",
      llvm::cl::desc("Data type to use for arithmetization."),
      llvm::cl::init(Integer),
      llvm::cl::values(
          clEnumVal(Integer, "decompose operations into 32 bit data types"))};
};

using CGGIPipelineBuilder =
    std::function<void(OpPassManager&, const MLIRToCGGIPipelineOptions&)>;

CGGIPipelineBuilder mlirToCGGIPipelineBuilder();

void mlirToCGGIPipeline(OpPassManager& pm,
                        const MLIRToCGGIPipelineOptions& options);
#endif

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

JaxiteBackendPipelineBuilder toJaxitePipelineBuilder();

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
