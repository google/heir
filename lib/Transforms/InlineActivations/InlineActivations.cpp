#include "lib/Transforms/InlineActivations/InlineActivations.h"

#include <cctype>

#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

#define DEBUG_TYPE "inline-activations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_INLINEACTIVATIONS
#include "lib/Transforms/InlineActivations/InlineActivations.h.inc"

struct InlineActivations : impl::InlineActivationsBase<InlineActivations> {
  using InlineActivationsBase::InlineActivationsBase;

  void runOnOperation() override {
    auto& symbolTable = getAnalysis<SymbolTable>();
    InlinerConfig config;

    auto module = getOperation();

    // Collect each of the direct function calls within the module.
    SmallVector<func::CallOp> callers;
    module->walk([&](func::CallOp caller) { callers.push_back(caller); });

    // Build the inliner interface.
    InlinerInterface inliner(&getContext());

    // Try to inline each of the call operations.
    for (auto caller : callers) {
      auto symbol = caller.getCalleeAttr();
      auto callee = symbolTable.lookup(symbol.getAttr());
      auto funcOp = dyn_cast_or_null<func::FuncOp>(callee);
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping " << symbol.getValue().str()
                                << " because it is not a function\n");

        continue;
      }

      if (funcOp.isDeclaration() || funcOp.isExternal()) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping " << symbol.getValue().str()
                                << " because it is not a private function\n");
        continue;
      }

      auto isReluWithDigits = [](StringRef name) {
        if (name == "relu") return true;
        if (name.starts_with("relu_")) {
          return llvm::all_of(name.drop_front(5), ::isdigit);
        }
        return false;
      };

      if (!isReluWithDigits(symbol.getValue())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping " << symbol.getValue().str()
                   << " because it is not an activation function\n");
        continue;
      }

      for (Operation& op : funcOp.getBody().getOps()) {
        for (auto& attr : caller->getAttrs()) {
          if (attr.getName() == SymbolTable::getSymbolAttrName() ||
              attr.getName() == SymbolTable::getVisibilityAttrName() ||
              attr.getName() == "callee") {
            continue;
          }
          op.setAttr(attr.getName(), attr.getValue());
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "- Inlining " << caller << "\n");
      if (failed(inlineCall(inliner, config.getCloneCallback(), caller, funcOp,
                            funcOp.getCallableRegion()))) {
        caller.emitError("function call cannot be inlined");
        return signalPassFailure();
      }
      caller.erase();
    }

    // After inlining, remove any now-dead functions
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createSymbolDCEPass());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
