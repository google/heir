#ifndef LIB_PIPELINES_YOSYSPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_YOSYSPIPELINEREGISTRATION_H_

#include <functional>
#include <string>

#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project

// The `mlir-to-cggi` pipeline, which booleanizes programs with Yosys and ABC.
// It is kept separate from the other boolean pipelines (see
// BooleanPipelineRegistration.h) so that code that does not need it, like the
// python bindings, does not depend on Yosys and ABC.

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

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_YOSYSPIPELINEREGISTRATION_H_
