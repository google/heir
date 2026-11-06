#ifndef LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_

#include <functional>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

// Pipelines lowering CGGI to its backends. The Yosys-based `mlir-to-cggi`
// pipeline lives in YosysPipelineRegistration.h.

namespace mlir::heir {

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
