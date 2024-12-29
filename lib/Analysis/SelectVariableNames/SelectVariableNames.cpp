#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

std::string suggestNameForValue(Value value) {
  auto suggestFunctions = {
      lwe::lweSuggestNameForType,
      openfhe::openfheSuggestNameForType,
  };
  // only the first suggestion is used
  // if no suggestion then do nothing
  for (auto suggest : suggestFunctions) {
    auto suggested = suggest(value.getType());
    if (!suggested.empty()) {
      return suggested;
    }
  }
  // default to v
  return "v";
}

SelectVariableNames::SelectVariableNames(Operation *op) {
  int i = 0;
  std::map<std::string, int> prefixCount;

  auto assignName = [&](Value value) {
    std::string name = suggestNameForValue(value);
    if (prefixCount.count(name) == 0) {
      // the first one is "prefix", the next on is "prefix1"
      prefixCount[name] = 1;
      variableNames.try_emplace(value, name);
    } else if (variableNames.count(value) == 0) {
      variableNames.try_emplace(value,
                                name + std::to_string(prefixCount[name]++));
    }
    // unique integer for each value
    variableToInteger.try_emplace(value, i++);
  };

  auto assignForOp = [&](Operation *op) {
    for (Value result : op->getResults()) {
      assignName(result);
    }

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Value arg : block.getArguments()) {
          assignName(arg);
        }
      }
    }
  };

  op->walk([&](func::FuncOp funcOp) {
    // clear the prefix count
    // different function has different name space
    prefixCount.clear();

    assignForOp(funcOp);
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      assignForOp(op);
      return WalkResult::advance();
    });
  });
}

}  // namespace heir
}  // namespace mlir
