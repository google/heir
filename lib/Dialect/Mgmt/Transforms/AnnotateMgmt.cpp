#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

#define GEN_PASS_DEF_ANNOTATEMGMT
#include "lib/Dialect/Mgmt/Transforms/Passes.h.inc"

void annotateMgmtAttr(Operation *top) {
  auto mergeIntoMgmtAttr = [&](Attribute levelAttr, Attribute dimensionAttr) {
    auto level = cast<IntegerAttr>(levelAttr).getInt();
    auto dimension = cast<IntegerAttr>(dimensionAttr).getInt();
    auto mgmtAttr = MgmtAttr::get(top->getContext(), level, dimension);
    return mgmtAttr;
  };
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    bool bodyContainsSecretGeneric = false;
    funcOp->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
      bodyContainsSecretGeneric = true;
      for (auto i = 0; i != genericOp.getBody()->getNumArguments(); ++i) {
        auto levelAttr = genericOp.removeArgAttr(i, "level");
        auto dimensionAttr = genericOp.removeArgAttr(i, "dimension");
        auto mgmtAttr = mergeIntoMgmtAttr(levelAttr, dimensionAttr);
        genericOp.setArgAttr(i, MgmtDialect::kArgMgmtAttrName, mgmtAttr);
      }

      genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }
        auto levelAttr = op->removeAttr("level");
        auto dimensionAttr = op->removeAttr("dimension");
        if (!levelAttr || !dimensionAttr) {
          return;
        }
        op->setAttr(MgmtDialect::kArgMgmtAttrName,
                    mergeIntoMgmtAttr(levelAttr, dimensionAttr));
      });
    });

    // handle generic-less function body
    // default to level 0 and dimension 2
    // otherwise secret-to-<scheme> won't find the mgmt attr
    if (!bodyContainsSecretGeneric) {
      for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
        auto argumentTy = funcOp.getFunctionType().getInput(i);
        if (isa<secret::SecretType>(argumentTy)) {
          funcOp.setArgAttr(i, MgmtDialect::kArgMgmtAttrName,
                            MgmtAttr::get(top->getContext(), 0, 2));
        }
      }
    }
  });
}

struct AnnotateMgmt : impl::AnnotateMgmtBase<AnnotateMgmt> {
  using AnnotateMgmtBase::AnnotateMgmtBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    annotateLevel(getOperation(), &solver);
    annotateDimension(getOperation(), &solver);
    // combine level and dimension into MgmtAttr
    // also removes the level/dimension annotations
    annotateMgmtAttr(getOperation());
  }
};

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
