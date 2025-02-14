#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
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

    auto secretArgAttr =
        StringAttr::get(ctx, secret::SecretDialect::kArgSecretAttrName);

    auto setSecretAttr = [&](func::FuncOp func) {
      for (unsigned i = 0; i < func.getNumArguments(); i++) {
        if (!isa<secret::SecretType>(func.getArgument(i).getType())) {
          func.setArgAttr(i, secretArgAttr, UnitAttr::get(ctx));
        }
      }
    };

    if (function.empty()) {
      module.walk([&](func::FuncOp func) { setSecretAttr(func); });
    } else {
      auto mainFunction = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(module, function));
      if (!mainFunction) {
        module.emitError("could not find function \"" + function +
                         "\" to secretize");
        signalPassFailure();
        return;
      }
      setSecretAttr(mainFunction);
    }
  }
};

}  // namespace heir
}  // namespace mlir
