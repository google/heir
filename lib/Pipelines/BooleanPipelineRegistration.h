#ifndef LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_

#include <string>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

namespace mlir::heir {

struct TosaToBooleanTfheOptions
    : public PassPipelineOptions<TosaToBooleanTfheOptions> {
  PassOptions::Option<bool> abcFast{*this, "abc-fast",
                                    llvm::cl::desc("Run abc in fast mode."),
                                    llvm::cl::init(false)};

  PassOptions::Option<int> unrollFactor{
      *this, "unroll-factor",
      llvm::cl::desc("Unroll loops by a given factor before optimizing. A "
                     "value of zero (default) prevents unrolling."),
      llvm::cl::init(0)};
};

struct TosaToBooleanJaxiteOptions : public TosaToBooleanTfheOptions {
  PassOptions::Option<int> parallelism{
      *this, "parallelism",
      llvm::cl::desc(
          "batching size for parallel execution on tpu. A value of 0 is no "
          "parallelism"),
      llvm::cl::init(0)};
};

void tosaToCGGIPipelineBuilder(OpPassManager &pm,
                               const TosaToBooleanTfheOptions &options,
                               const std::string &yosysFilesPath,
                               const std::string &abcPath,
                               bool abcBooleanGates);

void registerTosaToBooleanTfhePipeline(const std::string &yosysFilesPath,
                                       const std::string &abcPath);

void registerTosaToBooleanFpgaTfhePipeline(const std::string &yosysFilesPath,
                                           const std::string &abcPath);

void registerTosaToJaxitePipeline(const std::string &yosysFilesPath,
                                  const std::string &abcPath);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
