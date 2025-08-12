#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"

#include <map>
#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

std::string SelectVariableNames::suggestNameForValue(Value value) {
  if (auto opAsmTypeInterface =
          mlir::dyn_cast<OpAsmTypeInterface>(value.getType())) {
    std::string asmName;
    opAsmTypeInterface.getAsmName([&](StringRef name) { asmName = name; });
    return asmName;
  }
  return defaultPrefix;
}

SelectVariableNames::SelectVariableNames(Operation* op) {
  int i = 0;
  std::map<std::string, int> prefixCount;

  auto assignName = [&](Value value) {
    std::string name = suggestNameForValue(value);
    if (prefixCount.count(name) == 0 && name != defaultPrefix) {
      // for non-default prefix
      // the first one is "prefix", the next one is "prefix1"
      prefixCount[name] = 1;
      variableNames.try_emplace(value, name);
    } else if (variableNames.count(value) == 0) {
      if (prefixCount.count(name) == 0) {
        // for default prefix
        // the first one is "v0", the next one is "v1"
        prefixCount[name] = 0;
      }
      variableNames.try_emplace(value,
                                name + std::to_string(prefixCount[name]++));
    }
    // unique integer for each value
    variableToInteger.try_emplace(value, i++);
  };

  auto assignForOp = [&](Operation* op) {
    for (Value result : op->getResults()) {
      assignName(result);
    }

    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
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
    funcOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
      assignForOp(op);
      return WalkResult::advance();
    });
  });
}

}  // namespace heir
}  // namespace mlir
