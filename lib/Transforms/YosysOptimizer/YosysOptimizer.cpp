#include "include/Transforms/YosysOptimizer/YosysOptimizer.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "lib/Transforms/YosysOptimizer/LUTImporter.h"
#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"         // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "yosysoptimizer"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_YOSYSOPTIMIZER
#include "include/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

// $0: verilog filename
// $1: function name
// $2: yosys runfiles
// $3: abc path
// $4: abc fast option -fast
constexpr std::string_view kYosysTemplate = R"(
read_verilog {0};
hierarchy -check -top \{1};
proc; memory;
techmap -map {2}/techmap.v; opt;
abc -exe {3} -lut 3 {4};
opt_clean -purge;
rename -hide */c:*; rename -enumerate */c:*;
techmap -map {2}/map_lut_to_lut3.v; opt_clean -purge;
hierarchy -generate * o:Y i:*; opt; opt_clean -purge;
clean;
)";

struct YosysOptimizer : public impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;

  YosysOptimizer(std::string yosysFilesPath, std::string abcPath, bool abcFast)
      : yosysFilesPath(yosysFilesPath), abcPath(abcPath), abcFast(abcFast) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override;

 private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
};

// Globally optimize an MLIR module.
void YosysOptimizer::runOnOperation() {
  getOperation()->walk([&](func::FuncOp op) {
    // Translate function to Verilog. Translation will fail if the func
    // contains unsupported operations.
    // TODO(https://github.com/google/heir/issues/111): Directly convert MLIR to
    // Yosys' AST instead of using Verilog.
    char *filename = std::tmpnam(NULL);
    std::error_code EC;
    llvm::raw_fd_ostream of(filename, EC);
    if (failed(translateToVerilog(op, of)) || EC) {
      return WalkResult::interrupt();
    }
    of.close();

    // Invoke Yosys to translate to a combinational circuit and optimize.
    Yosys::yosys_setup();
    Yosys::log_error_stderr = true;
    LLVM_DEBUG(Yosys::log_streams.push_back(&std::cout));
    Yosys::run_pass(llvm::formatv(kYosysTemplate.data(), filename,
                                  op.getSymName(), yosysFilesPath, abcPath,
                                  abcFast ? "-fast" : ""));

    // Translate to MLIR and insert into the func
    std::stringstream cellOrder;
    Yosys::log_streams.push_back(&cellOrder);
    Yosys::run_pass("torder -stop * P*;");
    Yosys::log_streams.clear();
    auto topologicalOrder = getTopologicalOrder(cellOrder);

    // Insert the optimized MLIR.
    LUTImporter lutImporter = LUTImporter(&getContext());
    Yosys::RTLIL::Design *design = Yosys::yosys_get_design();
    func::FuncOp func =
        lutImporter.importModule(design->top_module(), topologicalOrder);
    op.getBody().takeBody(func.getBody());

    LLVM_DEBUG(llvm::dbgs()
               << "Converted & optimized func via yosys. Input func:\n"
               << op << "\n\nOutput func:\n"
               << func << "\n");

    return WalkResult::advance();
  });
  Yosys::yosys_shutdown();
}

std::unique_ptr<mlir::Pass> createYosysOptimizer(std::string yosysFilesPath,
                                                 std::string abcPath,
                                                 bool abcFast) {
  return std::make_unique<YosysOptimizer>(yosysFilesPath, abcPath, abcFast);
}

void registerYosysOptimizerPipeline(std::string yosysFilesPath,
                                    std::string abcPath) {
  PassPipelineRegistration<YosysOptimizerPipelineOptions>(
      "yosys-optimizer", "The yosys optimizer pipeline.",
      [yosysFilesPath, abcPath](OpPassManager &pm,
                                const YosysOptimizerPipelineOptions &options) {
        pm.addPass(
            createYosysOptimizer(yosysFilesPath, abcPath, options.abcFast));
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

}  // namespace heir
}  // namespace mlir
