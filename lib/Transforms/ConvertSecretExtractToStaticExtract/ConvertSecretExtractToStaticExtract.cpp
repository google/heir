#include "lib/Transforms/ConvertSecretExtractToStaticExtract/ConvertSecretExtractToStaticExtract.h"

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

#define GEN_PASS_DEF_CONVERTSECRETEXTRACTTOSTATICEXTRACT
#include "lib/Transforms/ConvertSecretExtractToStaticExtract/ConvertSecretExtractToStaticExtract.h.inc"

struct SecretExtractToStaticExtractConversion
    : OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

 public:
  SecretExtractToStaticExtractConversion(DataFlowSolver *solver,
                                         MLIRContext *context)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // TODO(#866): Add support for reads on multi-dimensional tensors
    auto index = extractOp->getOperand(1);

    auto *indexSecretnessLattice =
        solver->lookupState<SecretnessLattice>(index);

    // Implies that the tensor.extract op is newly created by this pass
    if (!indexSecretnessLattice) {
      extractOp->emitWarning()
          << "Secretness for tensor.extract index has not been set";
      return failure();
    }

    bool isIndexSecret = indexSecretnessLattice->getValue().getSecretness();

    // If index is not secret, no transformation is needed
    if (!isIndexSecret) return failure();

    ImplicitLocOpBuilder builder(extractOp->getLoc(), rewriter);

    auto argType = extractOp.getResult().getType();

    // Placeholder for extracted value
    auto dummyValue = builder.create<arith::ConstantOp>(
        argType, builder.getIntegerAttr(argType, 0));

    auto sizeAttr = extractOp->getAttrOfType<IntegerAttr>("size");

    // If size attribute is not provided, emit warning and return failure
    if (!sizeAttr) {
      extractOp->emitWarning()
          << "Cannot convert secret tensor.extract to static tensor.extract "
             "since a size attribute (`size`) has not "
             "been provided on the tensor.extract op:";
      return failure();
    }

    int size = sizeAttr.getInt();

    SmallVector<Value> iterArgs = {dummyValue};
    auto forOp = builder.create<affine::AffineForOp>(0, size, 1, iterArgs);

    builder.setInsertionPointToStart(forOp.getBody());

    // Check if the current index is equal to the secret index
    auto cond = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                              forOp.getInductionVar(), index);

    // Extract value at current index
    auto newExtractOp = builder.create<tensor::ExtractOp>(
        extractOp->getOperand(0), forOp.getInductionVar());

    auto ifOp = builder.create<scf::IfOp>(
        cond,
        [&](OpBuilder &b, Location loc) {
          // Yield value extracted at index
          b.create<scf::YieldOp>(loc, newExtractOp.getResult());
        },
        [&](OpBuilder &b, Location loc) {
          // Yield dummyValue
          b.create<scf::YieldOp>(loc, forOp.getRegionIterArgs().front());
        });

    // Create YieldOp for affine.for
    SmallVector<Value> results(ifOp->getOpResults());
    builder.create<affine::AffineYieldOp>(results);

    // Replace the old tensor.insert op with forOp's result
    rewriter.replaceOp(extractOp, forOp);

    return success();
  }

 private:
  DataFlowSolver *solver;
};

struct ConvertSecretExtractToStaticExtract
    : impl::ConvertSecretExtractToStaticExtractBase<
          ConvertSecretExtractToStaticExtract> {
  using ConvertSecretExtractToStaticExtractBase::
      ConvertSecretExtractToStaticExtractBase;

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

    patterns.add<SecretExtractToStaticExtractConversion>(&solver, context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
