#ifndef LIB_TARGET_SIMFHE_SIMFHEEMITTER_H_
#define LIB_TARGET_SIMFHE_SIMFHEEMITTER_H_

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Target/SimFHE/SimFHETemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace simfhe {

void registerToSimFHETranslation();

LogicalResult translateToSimFHE(Operation* op, llvm::raw_ostream& os);

/// SimFHEEmitter translates CKKS (e.g., the output of `--mlir-to-ckks`) to
/// Python code that can be used with SimFHE (https://github.com/bu-icsg/SimFHE)
/// in order to estimate the cost of the CKKS program.
///
/// How to use this:
// * Either use https://github.com/alexanderviand/SimFHE
//   or add "generated" to DECORATION_LIST in SimFHE's profiler.py
// * Run a HEIR/MLIR program through `heir-opt --mlir-to-ckks` or equivalent.
// * Run `heir-translate -emit-simfhe > generated.py` and place the resulting
// file in the SimFHE directory.
// * Running `python generated.py` in the SimFHE directory should (assuming
// you've installed the requirements)
//   should output the same kind of table one would get from running SimFHE's
//   `experiment.py`
//
// CAVEATS:
// SimFHEEmitter's codebase is somewhat rough around the edges and has little
// documentation, therefore it is not clear at all if the way the generated code
// uses SimFHE is correct. Interpreting SimFHE's output is also not
// straightforward, and therefore there is no guarantee that the SimFHE output
// is as expected, even for the `emit-simfhe.mlir` test. Finally, this emitter
// code is itself relatively rough and should not be relied upon too much.
//
// If you are interested in using the SimFHE emitter more seriously,
// it is highly recommended to reach out to the SimFHE authors first.
class SimFHEEmitter {
 public:
  SimFHEEmitter(llvm::raw_ostream& os, SelectVariableNames* variableNames);
  LogicalResult translate(Operation& operation);

 private:
  raw_indented_ostream os;
  SelectVariableNames* variableNames;

  LogicalResult printOperation(ModuleOp moduleOp);
  LogicalResult printOperation(func::FuncOp funcOp);
  LogicalResult printOperation(func::ReturnOp op);
  LogicalResult printOperation(ckks::AddOp op);
  LogicalResult printOperation(ckks::AddPlainOp op);
  LogicalResult printOperation(ckks::SubOp op);
  LogicalResult printOperation(ckks::SubPlainOp op);
  LogicalResult printOperation(ckks::MulOp op);
  LogicalResult printOperation(ckks::MulPlainOp op);
  LogicalResult printOperation(ckks::NegateOp op);
  LogicalResult printOperation(ckks::RotateOp op);
  LogicalResult printOperation(ckks::RelinearizeOp op);
  LogicalResult printOperation(ckks::RescaleOp op);
  LogicalResult printOperation(ckks::LevelReduceOp op);
  LogicalResult printOperation(ckks::BootstrapOp op);

  std::string getName(Value value) const {
    return variableNames->getNameForValue(value);
  }
};

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_SIMFHE_SIMFHEEMITTER_H_
