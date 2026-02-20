#include "lib/Transforms/CyclicRepetition/CyclicRepetition.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CYCLICREPETITION
#include "lib/Transforms/CyclicRepetition/CyclicRepetition.h.inc"

namespace {

struct CyclicRepetitionPattern : public OpRewritePattern<lwe::RLWEEncodeOp> {
  CyclicRepetitionPattern(MLIRContext* context, int64_t actualSlots)
      : OpRewritePattern<lwe::RLWEEncodeOp>(context),
        actualSlots(actualSlots) {}

  LogicalResult matchAndRewrite(lwe::RLWEEncodeOp op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getInput();
    auto tensorType = dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType) {
      return failure();
    }

    int64_t inputSize = tensorType.getDimSize(0);
    if (inputSize > actualSlots) {
      return failure();
    }

    int64_t numRepetitions = actualSlots / inputSize;
    if (numRepetitions <= 1) {
      return failure();
    }

    SmallVector<Value> tensorsToConcat;
    for (int64_t i = 0; i < numRepetitions; ++i) {
      tensorsToConcat.push_back(input);
    }

    auto concatOp =
        tensor::ConcatOp::create(rewriter, op.getLoc(), 0, tensorsToConcat);
    rewriter.modifyOpInPlace(
        op, [&]() { op.getInputMutable().assign(concatOp.getResult()); });
    return success();
  }

 private:
  int64_t actualSlots;
};

struct CyclicRepetition : impl::CyclicRepetitionBase<CyclicRepetition> {
  using CyclicRepetitionBase::CyclicRepetitionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto requestedAttr =
        module->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName);
    auto actualAttr =
        module->getAttrOfType<IntegerAttr>(kActualSlotCountAttrName);

    if (!requestedAttr || !actualAttr) {
      return;
    }

    int64_t requestedSlots = requestedAttr.getInt();
    int64_t actualSlots = actualAttr.getInt();

    if (actualSlots < requestedSlots) {
      module->emitOpError() << "Invalid acutalSlots = " << actualSlots
                            << " less than requestedSlots = " << requestedSlots;
      signalPassFailure();
      return;
    }

    if (actualSlots % requestedSlots != 0) {
      module->emitOpError()
          << "Invalid acutalSlots = " << actualSlots
          << " must be divisible by requestedSlots = " << requestedSlots;
      signalPassFailure();
      return;
    }

    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<CyclicRepetitionPattern>(context, actualSlots);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
