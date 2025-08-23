#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"

#include <cmath>
#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

#define DEBUG_TYPE "implement-rotate-and-reduce"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_IMPLEMENTROTATEANDREDUCE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

// TODO(#2136): Add a better way to test the correctness of this kernel.
LogicalResult convertRotateAndReduceOp(RotateAndReduceOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Converting tensor_ext.rotate_and_reduce op: "
                          << op << "\n");
  if (!op.getPlaintexts()) {
    // TODO(#2122): Implement the case where we accumulate the ciphertext slot
    // values.
    return op->emitOpError() << "rotate and reduce not implemented yet for "
                                "ciphertext value accumulation";
  }

  IRRewriter rewriter(op.getContext());
  TypedValue<RankedTensorType> input = op.getTensor();
  TypedValue<RankedTensorType> plaintexts = op.getPlaintexts();
  unsigned steps = op.getSteps().getZExtValue();
  unsigned period = op.getPeriod().getZExtValue();

  StringRef mulOpName = isa<IntegerType>(input.getType().getElementType())
                            ? "arith.muli"
                            : "arith.mulf";
  StringRef addOpName = isa<IntegerType>(input.getType().getElementType())
                            ? "arith.addi"
                            : "arith.addf";

  // Use a value of sqrt(n) as the baby step / giant step size.
  auto babySteps = static_cast<int64_t>(std::floor(std::sqrt(steps)));
  unsigned giantSteps = steps / babySteps;
  if (giantSteps * babySteps != steps) {
    return op.emitOpError()
           << "requires steps to be a multiple of sqrt(steps), but found "
              "steps="
           << steps << " and babySteps=" << babySteps;
  }
  LLVM_DEBUG(llvm::dbgs()
             << "Using baby-step / giant-step decomposition of sum of size "
             << steps << " with babySteps= " << babySteps
             << " and giantSteps= " << giantSteps << "\n");

  // Compute sqrt(n) ciphertext rotations of the input as baby-steps.
  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> babyStepVals;
  babyStepVals.push_back(input);
  for (int64_t i = 1; i < babySteps; ++i) {
    babyStepVals.push_back(rewriter
                               .create<tensor_ext::RotateOp>(
                                   op->getLoc(), input,
                                   rewriter.create<arith::ConstantIndexOp>(
                                       op->getLoc(), period * i))
                               .getResult());
  }

  unsigned plaintextSize = plaintexts.getType().getRank();
  SmallVector<OpFoldResult> offsets(plaintextSize, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sliceSizes;
  sliceSizes.reserve(plaintextSize);
  sliceSizes.push_back(rewriter.getIndexAttr(1));
  for (int64_t i = 1; i < plaintextSize; ++i) {
    sliceSizes.push_back(
        rewriter.getIndexAttr(plaintexts.getType().getDimSize(i)));
  }
  SmallVector<OpFoldResult> unitStrides(plaintextSize,
                                        rewriter.getIndexAttr(1));

  // Compute the inner baby step sums.
  Value result;
  for (unsigned k = 0; k < giantSteps; ++k) {
    Value innerSum;
    auto rotationIndex = rewriter.create<arith::ConstantIndexOp>(
        op->getLoc(), -babySteps * k * period);
    for (unsigned j = 0; j < babySteps; ++j) {
      offsets[0] = rewriter.getIndexAttr(j + k * babySteps * period);
      Value rotatedPlaintext = rewriter.create<tensor_ext::RotateOp>(
          op->getLoc(),
          rewriter.create<tensor::ExtractSliceOp>(op->getLoc(), input.getType(),
                                                  plaintexts, offsets,
                                                  sliceSizes, unitStrides),
          rotationIndex);
      Value multiplied =
          rewriter
              .create(OperationState(op->getLoc(), mulOpName,
                                     {rotatedPlaintext, babyStepVals[j]},
                                     {rotatedPlaintext.getType()}))
              ->getResults()[0];
      if (!innerSum) {
        innerSum = multiplied;
      } else {
        innerSum = rewriter
                       .create(OperationState(op->getLoc(), addOpName,
                                              {innerSum, multiplied},
                                              {innerSum.getType()}))
                       ->getResults()[0];
      }
    }

    auto rotatedSum = rewriter.create<tensor_ext::RotateOp>(
        op->getLoc(), innerSum,
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(),
                                                period * k * babySteps));
    if (!result) {
      result = rotatedSum;
    } else {
      result =
          rewriter
              .create(OperationState(op->getLoc(), addOpName,
                                     {result, rotatedSum}, {result.getType()}))
              ->getResults()[0];
    }
  }

  rewriter.replaceOp(op, result);
  return success();
}

struct ImplementRotateAndReduce
    : impl::ImplementRotateAndReduceBase<ImplementRotateAndReduce> {
  using ImplementRotateAndReduceBase::ImplementRotateAndReduceBase;

  void runOnOperation() override {
    getOperation()->walk([&](RotateAndReduceOp op) {
      if (failed(convertRotateAndReduceOp(op))) {
        op->emitOpError() << "failed to lower rotate_and_reduce op";
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
