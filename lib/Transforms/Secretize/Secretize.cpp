#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETIZE
#include "lib/Transforms/Secretize/Passes.h.inc"

struct Secretize : impl::SecretizeBase<Secretize> {
  using SecretizeBase::SecretizeBase;

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    ModuleOp module = getOperation();
    OpBuilder builder(module);

    auto mainFunction = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupSymbolIn(module, entryFunction));
    if (!mainFunction) {
      module.emitError("could not find entry point function");
      signalPassFailure();
      return;
    }

    auto secretArgAttr =
        StringAttr::get(ctx, secret::SecretDialect::kArgSecretAttrName);
    for (unsigned i = 0; i < mainFunction.getNumArguments(); i++) {
      mainFunction.setArgAttr(i, secretArgAttr, UnitAttr::get(ctx));
    }
  }
};

}  // namespace heir
}  // namespace mlir
