#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "convert-secret-insert-to-static-insert"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CONVERTSECRETINSERTTOSTATICINSERT
#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h.inc"

struct SecretInsertToStaticInsertConversion
    : OpRewritePattern<tensor::InsertOp> {
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;

 public:
  SecretInsertToStaticInsertConversion(DataFlowSolver *solver,
                                       MLIRContext *context)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(tensor::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    // TODO(#866): Add support for writes on multi-dimensional tensors
    if (insertOp.getDest().getType().getRank() != 1) {
      insertOp->emitWarning()
          << "Currently, transformation only supports 1D tensors:";
      return failure();
    }
    auto index = insertOp.getIndices().front();
    auto tensor = insertOp.getDest();
    auto insertedValue = insertOp.getScalar();

    auto *indexSecretnessLattice =
        solver->lookupState<SecretnessLattice>(index);

    // use secretness from lattice or, if no lattice found, set to unknown
    auto indexSecretness = indexSecretnessLattice
                               ? indexSecretnessLattice->getValue()
                               : Secretness();

    // If lattice is set to unknown,
    // apply transformation anyway but emit a warning
    if (!indexSecretness.isInitialized()) {
      insertOp->emitWarning()
          << "Secretness for tensor.insert index is unknown. "
             "Conservatively, the transformation will be applied:";
    }

    // If index is known to be public, no transformation is needed
    if (indexSecretness.isInitialized() && !indexSecretness.getSecretness())
      return failure();

    ImplicitLocOpBuilder builder(insertOp->getLoc(), rewriter);

    int size = insertOp.getDest().getType().getShape().front();

    SmallVector<Value> iterArgs = {tensor};
    auto forOp = builder.create<affine::AffineForOp>(0, size, 1, iterArgs);
    setValueToSecretness(solver, forOp.getInductionVar(), Secretness(false));

    builder.setInsertionPointToStart(forOp.getBody());

    // Check if the current index is equal to the secret index
    auto cond = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                              forOp.getInductionVar(), index);
    // Set secretness for cond
    for (auto result : cond->getResults()) {
      setValueToSecretness(solver, result, indexSecretness);
    }

    // Insert value at current index
    auto newInsertOp = builder.create<tensor::InsertOp>(
        insertedValue, forOp.getRegionIterArgs()[0], forOp.getInductionVar());
    // Set secretness for newInsertOp
    auto tensorSecretness =
        solver->getOrCreateState<SecretnessLattice>(tensor)->getValue();
    for (auto result : newInsertOp->getResults()) {
      setValueToSecretness(solver, result, tensorSecretness);
    }

    auto ifOp = builder.create<scf::IfOp>(
        cond,
        [&](OpBuilder &b, Location loc) {
          // Yield tensor with value inserted at index
          b.create<scf::YieldOp>(loc, newInsertOp.getResult());
        },
        [&](OpBuilder &b, Location loc) {
          // Yield old tensor
          b.create<scf::YieldOp>(loc, forOp.getRegionIterArgs().front());
        });
    // Set secretness for ifOp results: Combine indexSecretness with
    // tensorSecretness
    auto combinedSecretness =
        Secretness::combine({indexSecretness, tensorSecretness});
    for (auto result : ifOp->getResults()) {
      setValueToSecretness(solver, result, combinedSecretness);
    }

    // Create YieldOp for affine.for
    SmallVector<Value> results(ifOp->getOpResults());
    builder.create<affine::AffineYieldOp>(results);
    // Set secretness for forOp results
    for (auto result : forOp->getResults()) {
      setValueToSecretness(solver, result, combinedSecretness);
    }

    // Replace the old tensor.insert op with forOp's result
    rewriter.replaceOp(insertOp, forOp);

    return success();
  }

 private:
  DataFlowSolver *solver;
  static inline void setValueToSecretness(DataFlowSolver *solver, Value value,
                                          Secretness secretness) {
    auto *lattice = solver->getOrCreateState<SecretnessLattice>(value);
    solver->propagateIfChanged(lattice, lattice->join(secretness));
  }
};

struct ConvertSecretInsertToStaticInsert
    : impl::ConvertSecretInsertToStaticInsertBase<
          ConvertSecretInsertToStaticInsert> {
  using ConvertSecretInsertToStaticInsertBase::
      ConvertSecretInsertToStaticInsertBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    patterns.add<SecretInsertToStaticInsertConversion>(&solver, context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    LLVM_DEBUG({
      // Add an attribute to the operations to show determined secretness
      OpBuilder builder(context);
      getOperation()->walk([&](Operation *op) {
        if (op->getNumResults() == 0) return;
        auto *secretnessLattice =
            solver.lookupState<SecretnessLattice>(op->getResult(0));
        if (!secretnessLattice) {
          op->setAttr("secretness", builder.getStringAttr("null"));
          return;
        }
        if (!secretnessLattice->getValue().isInitialized()) {
          op->setAttr("secretness", builder.getStringAttr("unknown"));
          return;
        }
        op->setAttr(
            "secretness",
            builder.getBoolAttr(secretnessLattice->getValue().getSecretness()));
        return;
      });
    });
  }
};

}  // namespace heir
}  // namespace mlir
