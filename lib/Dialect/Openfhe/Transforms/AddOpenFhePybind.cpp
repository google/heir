#include "lib/Dialect/Openfhe/Transforms/AddOpenFhePybind.h"

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_ADDOPENFHEPYBIND
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct AddOpenFhePybind : impl::AddOpenFhePybindBase<AddOpenFhePybind> {
  using AddOpenFhePybindBase::AddOpenFhePybindBase;

  void runOnOperation() override {
    ModuleOp topModule = getOperation();

    // 1. Create the nested module
    OpBuilder builder(topModule.getBodyRegion());
    // Insert at the end of the top-level module
    builder.setInsertionPointToEnd(topModule.getBody());

    auto nestedModule =
        ModuleOp::create(topModule.getLoc(), StringRef("pybind_bindings"));
    builder.insert(nestedModule);

    // 2. Tag it with heir.pybind_module (UnitAttr)
    nestedModule->setAttr("heir.pybind_module", builder.getUnitAttr());

    // 3. Attach metadata attributes
    if (!pybindModuleName.empty()) {
      nestedModule->setAttr("pybind.module_name",
                            builder.getStringAttr(pybindModuleName));
    }
    if (!pybindImports.empty()) {
      SmallVector<Attribute> importAttrs;
      for (const auto& import : pybindImports) {
        importAttrs.push_back(builder.getStringAttr(import));
      }
      nestedModule->setAttr("pybind.imports",
                            builder.getArrayAttr(importAttrs));
    }

    // 4. Walk all public functions in the outer module and clone their
    // signatures into the nested module.
    OpBuilder nestedBuilder(nestedModule.getBodyRegion());

    for (auto funcOp : topModule.getOps<func::FuncOp>()) {
      // Skip private/nested functions (we only want public top-level functions)
      if (funcOp.isPrivate()) continue;

      // Create the private function declaration in the nested module
      auto declOp = func::FuncOp::create(funcOp.getLoc(), funcOp.getName(),
                                         funcOp.getFunctionType());
      declOp.setPrivate();

      // If the function returns a shaped type, attach its shape as an attribute
      if (funcOp.getNumResults() > 0) {
        if (auto shapedTy = dyn_cast<ShapedType>(funcOp.getResultTypes()[0])) {
          if (shapedTy.hasRank()) {
            auto shape = shapedTy.getShape();
            declOp->setAttr("pybind.return_shape",
                            builder.getI64ArrayAttr(shape));
          }
        }
      }

      nestedBuilder.insert(declOp);
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
