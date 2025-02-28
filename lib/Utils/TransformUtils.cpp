#include "lib/Utils/TransformUtils.h"

#include <set>
#include <string>
#include <string_view>

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project

namespace mlir {
namespace heir {

func::FuncOp detectEntryFunction(ModuleOp moduleOp,
                                 std::string_view entryFunction) {
  // get from user input
  auto entryFunc = moduleOp.lookupSymbol<func::FuncOp>(entryFunction);
  if (!entryFunc) {
    // detect the entry function with the following heuristic:
    // 1. the function name does not contain "__"
    // 2. the function is not a declaration
    // 3. the function is not called by any other function
    // 4. the first function that satisfies the above conditions

    // get all the called functions
    std::set<std::string> calledFuncs;
    moduleOp->walk<WalkOrder::PreOrder>([&](func::CallOp callOp) {
      auto callee = callOp.getCallee();
      calledFuncs.insert(std::string(callee));
    });

    moduleOp->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      auto funcSymName = funcOp.getSymName();
      if (funcSymName.find("__") != std::string::npos ||
          calledFuncs.find(std::string(funcSymName)) != calledFuncs.end() ||
          funcOp.isDeclaration()) {
        return WalkResult::advance();
      }
      entryFunc = funcOp;
      return WalkResult::interrupt();
    });
  }
  // still no result then emit warning
  if (!entryFunc) {
    moduleOp->emitWarning(
        "Entry function not found, please provide entry-function in the pass "
        "options");
  }
  return entryFunc;
}

}  // namespace heir
}  // namespace mlir
