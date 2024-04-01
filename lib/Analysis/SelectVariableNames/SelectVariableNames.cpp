#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"

#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"   // from @llvm-project

namespace mlir {
namespace heir {

SelectVariableNames::SelectVariableNames(Operation *op) {
  int i = 0;
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Value result : op->getResults()) {
      variableNames.try_emplace(result, i++);
    }

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Value arg : block.getArguments()) {
          variableNames.try_emplace(arg, i++);
        }
      }
    }

    return WalkResult::advance();
  });
}

}  // namespace heir
}  // namespace mlir
