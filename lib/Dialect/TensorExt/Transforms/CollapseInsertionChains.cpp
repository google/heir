#include "lib/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "collapse-insertion-chains"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_COLLAPSEINSERTIONCHAINS
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

/// A pattern that searches for sequences of extract + insert, where the
/// indices extracted and inserted have the same offset, and replaced them with
/// a single rotate operation.
struct ConvertAlignedExtractInsertToRotate
    : public OpRewritePattern<tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  // Given an insert and extract op, compute the shift between their two
  // access indices. Only works for 1D tensors.
  FailureOr<int64_t> calculateShift(tensor::InsertOp insertOp,
                                    tensor::ExtractOp extractOp) const {
    auto insertIndexRes = get1DExtractionIndex(insertOp);
    auto extractionIndexRes = get1DExtractionIndex(extractOp);
    if (failed(insertIndexRes) || failed(extractionIndexRes)) return failure();

    auto insertTensorType =
        mlir::cast<RankedTensorType>(insertOp.getDest().getType());
    auto extractTensorType =
        mlir::cast<RankedTensorType>(extractOp.getTensor().getType());
    if (insertTensorType.getShape() != extractTensorType.getShape())
      return failure();

    auto shift = (extractionIndexRes.value() - insertIndexRes.value());
    if (shift < 0) shift += insertTensorType.getShape()[0];

    LLVM_DEBUG({
      llvm::dbgs() << "Insertion index: " << insertIndexRes.value() << "\n";
      llvm::dbgs() << "Extraction index: " << extractionIndexRes.value()
                   << "\n";
      llvm::dbgs() << "Shift: " << shift << "\n";
    });
    return shift;
  }

  LogicalResult matchAndRewrite(tensor::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Visiting insert op: " << insertOp << "\n");
    auto extractOp = insertOp.getScalar().getDefiningOp<tensor::ExtractOp>();
    if (!extractOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Insert op " << insertOp
                 << " is not directly operating on an extract op.\n");
      return failure();
    }

    auto shiftRes = calculateShift(insertOp, extractOp);
    if (failed(shiftRes)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Insert op " << insertOp << " has incalculable shift\n");
      return failure();
    }

    int64_t shift = shiftRes.value();
    DenseSet<int64_t> accessedIndices;
    accessedIndices.insert(get1DExtractionIndex(insertOp).value());

    // Check if there are corresponding insertions into all other indices,
    // which are extracted from the same source tensor with the same shift.
    //
    // The problem is that because tensors have value semantics and not pointer
    // semantics, we will see each new insertion use a different SSA value, like
    // this:
    //
    // %extracted_1 = tensor.extract %original_source[%c1] : tensor<4096xi16>
    // %inserted_1 = tensor.insert %extracted_1 into %original_dest[%c5] :
    // tensor<4096xi16> %extracted_2 = tensor.extract %original_source[%c2] :
    // tensor<4096xi16> %inserted_2 = tensor.insert %extracted_2 into
    // %inserted_1[%c6] : tensor<4096xi16>
    //
    // Note how inserted_1 replaces original_dest for the subsequent insert,
    // and inserted_2 will replace inserted_1 for the next one.
    //
    // So we need to traverse the insertions in order to follow the chain. Note
    // that a more sophisticated pass might be able to support less
    // well-structured DAGs of insertions and extractions, but we will improve
    // this when that becomes necessary.

    // Also note that the greedy pattern rewriter will start from the first op
    // in the block order, so we can assume that if the pattern matches, it
    // matches on the first insert op encountered.
    auto extractionSource = extractOp.getTensor();
    auto current = insertOp;
    while (true) {
      bool foundNext = false;

      for (auto *user : current.getResult().getUsers()) {
        if (auto nextInsert = dyn_cast<tensor::InsertOp>(user)) {
          auto nextExtract =
              nextInsert.getScalar().getDefiningOp<tensor::ExtractOp>();
          if (!nextExtract) continue;

          // We're inserting into this tensor from a different tensor than
          // earlier insertions in the chain, so we can't continue.
          if (nextExtract.getTensor() != extractionSource) return failure();

          auto nextShiftRes = calculateShift(nextInsert, nextExtract);
          if (failed(nextShiftRes)) return failure();

          if (nextShiftRes.value() != shift) return failure();

          accessedIndices.insert(get1DExtractionIndex(nextInsert).value());
          current = nextInsert;
          foundNext = true;
        }
      }

      // This can either be because we reached the end of the chain, or else
      // because the chain is incomplete.
      if (!foundNext) break;
    }

    LLVM_DEBUG({
      auto vec = llvm::to_vector(accessedIndices);
      llvm::sort(vec);
      llvm::dbgs() << "Chain ending in insert op " << insertOp
                   << " has access indices: [";
      for (auto index : vec) {
        llvm::dbgs() << index << ", ";
      }
      llvm::dbgs() << "]\n";
    });

    // We didn't cover the entire tensor, so the downstream user of this tensor
    // may be depending on the original data in the untouched indices being in
    // tact.
    auto tensorSize =
        mlir::cast<RankedTensorType>(extractionSource.getType()).getShape()[0];
    if (accessedIndices.size() != tensorSize) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Chain has only " << accessedIndices.size()
                 << " accessed indices, but tensor has " << tensorSize << "\n");
      return failure();
    }

    // The last insertion must be replaced because its user is the final end
    // user
    rewriter.replaceOpWithNewOp<tensor_ext::RotateOp>(
        current, extractionSource,
        arith::ConstantIndexOp::create(rewriter, current.getLoc(), shift));

    // The rest of the chain of insertions and extractions itself will be
    // DCE'd by canonicalization if possible.
    return success();
  }
};

struct CollapseInsertionChains
    : impl::CollapseInsertionChainsBase<CollapseInsertionChains> {
  using CollapseInsertionChainsBase::CollapseInsertionChainsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConvertAlignedExtractInsertToRotate>(context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
