#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
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
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
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
        auto levelAttr = genericOp.removeOperandAttr(i, "level");
        auto dimensionAttr = genericOp.removeOperandAttr(i, "dimension");
        auto mgmtAttr = mergeIntoMgmtAttr(levelAttr, dimensionAttr);
        genericOp.setOperandAttr(i, MgmtDialect::kArgMgmtAttrName, mgmtAttr);
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

void copyMgmtAttrToPlaintextOperand(Operation *top) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    // insert before the generic op
    auto funcOp = genericOp->getParentOfType<func::FuncOp>();
    OpBuilder b = OpBuilder::atBlockBegin(&funcOp.getBody().front());
    b.setInsertionPoint(genericOp);

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (mlir::isa<arith::MulIOp, arith::AddIOp, arith::SubIOp, arith::MulFOp,
                    arith::AddFOp, arith::SubFOp>(op)) {
        MgmtAttr mgmtAttr;
        int plaintextOperandIndex = -1;
        // find the plaintext operand and get mgmt attr from the other operand
        for (auto i = 0; i != op->getNumOperands(); ++i) {
          auto attr = findMgmtAttrAssociatedWith(op->getOperand(i));
          if (!attr) {
            plaintextOperandIndex = i;
          } else {
            mgmtAttr = attr;
          }
        }
        if (plaintextOperandIndex == -1) {
          return;
        }

        assert(mgmtAttr &&
               "ct-pt op should have at least one ciphertext operand");
        auto plaintextOperand = op->getOperand(plaintextOperandIndex);
        // create a new mgmt.no_op with mgmt attr
        // this is because plaintextOperand can be used in multiple places
        // and we don't want to change the original one
        if (auto *definingOp = plaintextOperand.getDefiningOp()) {
          b.setInsertionPointAfter(definingOp);
        }
        auto newNoOp = b.create<mgmt::NoOp>(
            op->getLoc(), plaintextOperand.getType(), plaintextOperand);
        newNoOp->setAttr(MgmtDialect::kArgMgmtAttrName, mgmtAttr);
        op->setOperand(plaintextOperandIndex, newNoOp.getResult());
      }
    });
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

    annotateLevel(getOperation(), &solver, baseLevel);
    annotateDimension(getOperation(), &solver);
    // combine level and dimension into MgmtAttr
    // also removes the level/dimension annotations
    annotateMgmtAttr(getOperation());
    // annotate the plaintext operand of ct-pt operations
    copyMgmtAttrToPlaintextOperand(getOperation());
  }
};

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
