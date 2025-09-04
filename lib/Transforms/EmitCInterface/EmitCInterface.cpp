#include "lib/Transforms/EmitCInterface/EmitCInterface.h"

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_EMITCINTERFACE
#include "lib/Transforms/EmitCInterface/EmitCInterface.h.inc"

struct EmitCInterface : impl::EmitCInterfaceBase<EmitCInterface> {
  using EmitCInterfaceBase::EmitCInterfaceBase;

  void runOnOperation() override {
    getOperation()->walk([&](mlir::func::FuncOp op) {
      if (op.isPublic()) {
        op->setAttr("llvm.emit_c_interface",
                    mlir::UnitAttr::get(&getContext()));
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
