#include "lib/Transforms/RemoveUnusedPureCall/RemoveUnusedPureCall.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_REMOVEUNUSEDPURECALL
#include "lib/Transforms/RemoveUnusedPureCall/RemoveUnusedPureCall.h.inc"

struct RemoveUnusedPureCall
    : public impl::RemoveUnusedPureCallBase<RemoveUnusedPureCall> {
  using RemoveUnusedPureCallBase::RemoveUnusedPureCallBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbolTables;
    module.walk([&](func::CallOp call) {
      if (!call.use_empty()) return;
      auto callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isDeclaration() || !isClientHelper(callee)) return;
      if (llvm::all_of(callee.getBody().getOps(),
                       [](Operation& op) { return isMemoryEffectFree(&op); }))
        call.erase();
    });
  }
};

}  // namespace heir
}  // namespace mlir
