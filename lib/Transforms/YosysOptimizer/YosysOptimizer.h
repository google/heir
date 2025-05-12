#ifndef LIB_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_
#define LIB_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_

#include <memory>
#include <string>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project

namespace mlir {
namespace heir {

enum Mode { Boolean, LUT };

std::unique_ptr<mlir::Pass> createYosysOptimizer(
    const std::string &yosysFilesPath, const std::string &abcPath, bool abcFast,
    int unrollFactor = 0, Mode mode = LUT, bool printStats = false);

#define GEN_PASS_DECL
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

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

  PassOptions::Option<enum Mode> mode{
      *this, "mode",
      llvm::cl::desc("Map gates to boolean gates or lookup table gates."),
      llvm::cl::init(LUT),
      llvm::cl::values(clEnumVal(Boolean, "use boolean gates"),
                       clEnumVal(LUT, "use lookup tables"))};

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

#endif  // LIB_TRANSFORMS_YOSYSOPTIMIZER_YOSYSOPTIMIZER_H_
