#ifndef INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_
#define INCLUDE_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

std::unique_ptr<mlir::Pass> createYosysOptimizer(
    const std::string &yosysFilesPath, const std::string &abcPath, bool abcFast,
    int unrollFactor = 0, bool printStats = false);

#define GEN_PASS_DECL
#include "include/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

struct YosysOptimizerPipelineOptions
    : public PassPipelineOptions<YosysOptimizerPipelineOptions> {
  PassOptions::Option<bool> abcFast{*this, "abc-fast",
                                    llvm::cl::desc("Run abc in fast mode."),
                                    llvm::cl::init(false)};

  PassOptions::Option<int> unrollFactor{
      *this, "unroll-factor",
      llvm::cl::desc("Unroll loops by a given factor before optimizing. A "
                     "value of zero (default) prevents unrolling."),
      llvm::cl::init(0)};

  PassOptions::Option<bool> printStats{
      *this, "print-stats",
      llvm::cl::desc("Prints statistics about the optimized circuit"),
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
