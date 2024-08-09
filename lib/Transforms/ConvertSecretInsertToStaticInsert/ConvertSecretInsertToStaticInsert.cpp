#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

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
    auto index = insertOp->getOperand(2);
    auto tensor = insertOp->getOperand(1);
    auto insertedValue = insertOp->getOperand(0);

    auto *indexSecretnessLattice =
        solver->lookupState<SecretnessLattice>(index);

    // Implies that the tensor.insert op is newly created by this pass
    if (!indexSecretnessLattice) {
      insertOp->emitWarning()
          << "Secretness for tensor.insert index has not been set";
      return failure();
    }

    bool isIndexSecret = indexSecretnessLattice->getValue().getSecretness();

    // If index is not secret, no transformation is needed
    if (!isIndexSecret) return failure();

    ImplicitLocOpBuilder builder(insertOp->getLoc(), rewriter);

    auto sizeAttr = insertOp->getAttrOfType<IntegerAttr>("size");

    // If size attribute is not provided, emit warning and return failure
    if (!sizeAttr) {
      insertOp->emitWarning()
          << "Cannot convert secret tensor.insert to static tensor.insert "
             "since a size attribute (`size`) has not "
             "been provided on the tensor.insert op:";
      return failure();
    }

    int size = sizeAttr.getInt();

    SmallVector<Value> iterArgs = {tensor};
    auto forOp = builder.create<affine::AffineForOp>(0, size, 1, iterArgs);

    builder.setInsertionPointToStart(forOp.getBody());

    // Check if the current index is equal to the secret index
    auto cond = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                              forOp.getInductionVar(), index);

    // Insert value at current index
    auto newInsertOp = builder.create<tensor::InsertOp>(
        insertedValue, tensor, forOp.getInductionVar());

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

    // Create YieldOp for affine.for
    SmallVector<Value> results(ifOp->getOpResults());
    builder.create<affine::AffineYieldOp>(results);

    // Replace the old tensor.insert op with forOp's result
    rewriter.replaceOp(insertOp, forOp);

    return success();
  }

 private:
  DataFlowSolver *solver;
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
  }
};

}  // namespace heir
}  // namespace mlir
