#include "lib/Transforms/ApplyFolders/ApplyFolders.h"

#include <cstdint>
#include <utility>

#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_APPLYFOLDERS
#include "lib/Transforms/ApplyFolders/ApplyFolders.h.inc"

namespace {

// keep in anonymous namespace
#include "lib/Transforms/ApplyFolders/Patterns.cpp.inc"

}  // namespace

struct ApplyFolders : impl::ApplyFoldersBase<ApplyFolders> {
  using ApplyFoldersBase::ApplyFoldersBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    constexpr int64_t kMaxConstantFoldElements = 1 << 16;
    tensor::ControlConstantExtractSliceFusionFn controlFn =
        [](tensor::ExtractSliceOp op) {
          DenseElementsAttr cst;
          if (matchPattern(op.getSource(), m_Constant(&cst))) {
            if (auto shaped = dyn_cast<ShapedType>(cst.getType()))
              if (shaped.getNumElements() > kMaxConstantFoldElements)
                return false;
          }
          return true;
        };
    tensor::populateFoldConstantExtractSlicePatterns(patterns, controlFn);
    tensor::populateFoldTensorSubsetOpPatterns(patterns);
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateDropRedundantInsertSliceRankExpansionPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    populateWithGenerated(patterns);
    // Use the greedy pattern driver to apply folders.
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
