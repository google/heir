#include "lib/Transforms/InlineActivations/InlineActivations.h"

#include <map>
#include <string>

#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"  // from @llvm-project

#define DEBUG_TYPE "inline-activations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_INLINEACTIVATIONS
#include "lib/Transforms/InlineActivations/InlineActivations.h.inc"

namespace {

static std::map<std::string, bool> activationMap = {
    {"relu", true},
};

}  // namespace

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
      auto funcOp = dyn_cast<func::FuncOp>(callee);
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

      if (!activationMap.contains(symbol.getValue().str())) {
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
                            funcOp.getCallableRegion())))
        caller.emitError("function call cannot be inlined");

      caller.erase();
      if (funcOp.use_empty()) funcOp.erase();
    }
  }
};

}  // namespace heir
}  // namespace mlir
