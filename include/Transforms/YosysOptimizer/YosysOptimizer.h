#ifndef INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_
#define INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

std::unique_ptr<mlir::Pass> createYosysOptimizer(
    const std::string &yosysFilesPath, const std::string &abcPath,
    bool abcFast);

#define GEN_PASS_DECL
#include "include/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

struct YosysOptimizerPipelineOptions
    : public PassPipelineOptions<YosysOptimizerPipelineOptions> {
  PassOptions::Option<bool> abcFast{*this, "abc-fast",
                                    llvm::cl::desc("Run abc in fast mode."),
                                    llvm::cl::init(false)};
};

// registerYosysOptimizerPipeline registers a Yosys pipeline pass using
// runfiles, the location of Yosys techlib files, and abcPath, the location of
// the abc binary.
void registerYosysOptimizerPipeline(const std::string &yosysFilesPath,
                                    const std::string &abcPath);

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_
