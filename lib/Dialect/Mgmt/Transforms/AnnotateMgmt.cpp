#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

#define GEN_PASS_DEF_ANNOTATEMGMT
#include "lib/Dialect/Mgmt/Transforms/Passes.h.inc"

void annotateMgmtAttr(Operation *top) {
  auto mergeIntoMgmtAttr = [&](Attribute levelAttr, Attribute dimensionAttr,
                               Attribute scaleAttr) {
    auto level = cast<IntegerAttr>(levelAttr).getInt();
    auto dimension = cast<IntegerAttr>(dimensionAttr).getInt();
    if (!scaleAttr) {
      return MgmtAttr::get(top->getContext(), level, dimension);
    }
    auto scale = cast<IntegerAttr>(scaleAttr).getInt();
    return MgmtAttr::get(top->getContext(), level, dimension, scale);
  };
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    // handle plaintext
    funcOp->walk<WalkOrder::PreOrder>([&](mgmt::InitOp initOp) {
      auto levelAttr = initOp->removeAttr(kArgLevelAttrName);
      auto dimensionAttr = initOp->removeAttr(kArgDimensionAttrName);
      if (!levelAttr || !dimensionAttr) {
        return;
      }
      auto scaleAttr = initOp->removeAttr(kArgScaleAttrName);
      auto mgmtAttr = mergeIntoMgmtAttr(levelAttr, dimensionAttr, scaleAttr);
      initOp->setAttr(MgmtDialect::kArgMgmtAttrName, mgmtAttr);
    });

    bool bodyContainsSecretGeneric = false;
    funcOp->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
      bodyContainsSecretGeneric = true;
      for (auto i = 0; i != genericOp.getBody()->getNumArguments(); ++i) {
        auto levelAttr = genericOp.removeOperandAttr(i, kArgLevelAttrName);
        auto dimensionAttr =
            genericOp.removeOperandAttr(i, kArgDimensionAttrName);
        auto scaleAttr = genericOp.removeOperandAttr(i, kArgScaleAttrName);
        auto mgmtAttr = mergeIntoMgmtAttr(levelAttr, dimensionAttr, scaleAttr);
        genericOp.setOperandAttr(i, MgmtDialect::kArgMgmtAttrName, mgmtAttr);
      }

      genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }
        auto levelAttr = op->removeAttr(kArgLevelAttrName);
        auto dimensionAttr = op->removeAttr(kArgDimensionAttrName);
        if (!levelAttr || !dimensionAttr) {
          return;
        }
        auto scaleAttr = op->removeAttr(kArgScaleAttrName);
        op->setAttr(MgmtDialect::kArgMgmtAttrName,
                    mergeIntoMgmtAttr(levelAttr, dimensionAttr, scaleAttr));
      });

      // Add yielded result as attribute on secret.generic
      secret::YieldOp yieldOp = genericOp.getYieldOp();
      for (auto &opOperand : yieldOp->getOpOperands()) {
        FailureOr<Attribute> attrResult = findAttributeAssociatedWith(
            opOperand.get(), MgmtDialect::kArgMgmtAttrName);

        if (failed(attrResult)) continue;
        genericOp.setResultAttr(opOperand.getOperandNumber(),
                                MgmtDialect::kArgMgmtAttrName,
                                attrResult.value());
      }
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

  populateOperandAttrInterface(top, MgmtDialect::kArgMgmtAttrName);
}

struct AnnotateMgmt : impl::AnnotateMgmtBase<AnnotateMgmt> {
  using AnnotateMgmtBase::AnnotateMgmtBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<LevelAnalysisBackward>(symbolTable);
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    annotateLevel(getOperation(), &solver, baseLevel);
    annotateDimension(getOperation(), &solver);
    // Combine level and dimension (and optional scale) into MgmtAttr
    // also removes the level/dimension/(optional scale) annotations.
    // The optional scale is passed by annotateScale() when calling
    // this pass.
    annotateMgmtAttr(getOperation());
  }
};

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
