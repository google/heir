#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"

#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project

namespace mlir {
namespace heir {

SelectVariableNames::SelectVariableNames(Operation *op) {
  int i = 0;
  std::string prefix = "v";
  op->walk([&](Operation *op) {
    return llvm::TypeSwitch<Operation &, WalkResult>(*op)
        // Function arguments need names
        .Case<func::FuncOp>([&](auto op) {
          for (Value arg : op.getArguments()) {
            variableNames.try_emplace(arg, prefix + std::to_string(i++));
          }
          return WalkResult::advance();
        })
        // Operation results need names
        .Default([&](Operation &) {
          for (Value result : op->getResults()) {
            variableNames.try_emplace(result, prefix + std::to_string(i++));
          }
          return WalkResult::advance();
        });
    // TODO(https://github.com/google/heir/issues/229): handle block arguments
  });
}

}  // namespace heir
}  // namespace mlir
