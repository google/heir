#include "lib/Dialect/Debug/Transforms/InsertLinalgCheckpoints.h"

#include <iterator>
#include <string>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "llvm/include/llvm/ADT/DenseSet.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project

namespace mlir {
namespace heir {
namespace debug {

#define GEN_PASS_DEF_DEBUGINSERTLINALGCHECKPOINTS
#include "lib/Dialect/Debug/Transforms/Passes.h.inc"

namespace {

std::string inputCheckpointName(func::FuncOp funcOp, unsigned argumentIndex) {
  return (funcOp.getSymName() + "/input/" + std::to_string(argumentIndex))
      .str();
}

std::string resultCheckpointName(func::FuncOp funcOp, Operation* op,
                                 unsigned operationIndex,
                                 unsigned resultIndex) {
  return (funcOp.getSymName() + "/" + op->getName().getStringRef() + "/" +
          std::to_string(operationIndex) + "/" + std::to_string(resultIndex))
      .str();
}

struct InsertLinalgCheckpoints
    : impl::DebugInsertLinalgCheckpointsBase<InsertLinalgCheckpoints> {
  using DebugInsertLinalgCheckpointsBase::DebugInsertLinalgCheckpointsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      module.emitError("failed to run the secretness analysis");
      return signalPassFailure();
    }

    llvm::DenseSet<Value> validatedValues;
    module.walk(
        [&](debug::ValidateOp op) { validatedValues.insert(op.getInput()); });

    bool foundEntryFunction = entryFunction.empty();
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isExternal() ||
          (!entryFunction.empty() && funcOp.getSymName() != entryFunction)) {
        continue;
      }
      foundEntryFunction = true;
      instrumentFunction(funcOp, solver, validatedValues);
    }

    if (!foundEntryFunction) {
      module.emitError() << "entry function '" << entryFunction
                         << "' was not found";
      signalPassFailure();
    }
  }

 private:
  void instrumentFunction(func::FuncOp funcOp, DataFlowSolver& solver,
                          llvm::DenseSet<Value>& validatedValues) {
    if (includeInputs) {
      OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
      for (auto [index, argument] : llvm::enumerate(funcOp.getArguments())) {
        if (!isSecret(argument, &solver) || validatedValues.contains(argument))
          continue;
        debug::ValidateOp::create(builder, argument.getLoc(), argument,
                                  inputCheckpointName(funcOp, index), nullptr);
        validatedValues.insert(argument);
      }
    }

    SmallVector<Operation*> linalgOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (op->getParentOfType<func::FuncOp>() == funcOp)
        linalgOps.push_back(op.getOperation());
    });

    for (auto [operationIndex, op] : llvm::enumerate(linalgOps)) {
      OpBuilder builder(op->getBlock(), std::next(op->getIterator()));
      for (auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
        if (!isSecret(result, &solver) || validatedValues.contains(result))
          continue;
        debug::ValidateOp::create(
            builder, result.getLoc(), result,
            resultCheckpointName(funcOp, op, operationIndex, resultIndex),
            nullptr);
        validatedValues.insert(result);
      }
    }
  }
};

}  // namespace
}  // namespace debug
}  // namespace heir
}  // namespace mlir
