#include "lib/Transforms/RemoveUnusedPureCall/RemoveUnusedPureCall.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_REMOVEUNUSEDPURECALL
#include "lib/Transforms/RemoveUnusedPureCall/RemoveUnusedPureCall.h.inc"

struct RemoveUnusedPureCall
    : public impl::RemoveUnusedPureCallBase<RemoveUnusedPureCall> {
  using RemoveUnusedPureCallBase::RemoveUnusedPureCallBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DenseSet<StringAttr> pureFunctions;

    module.walk([&](func::FuncOp func) {
      if (isClientHelper(func.getOperation())) {
        pureFunctions.insert(func.getSymNameAttr());
      }
    });

    module.walk([&](func::CallOp call) {
      if (call.use_empty() &&
          pureFunctions.contains(call.getCalleeAttr().getAttr())) {
        call.erase();
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
