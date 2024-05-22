#include "lib/Dialect/Polynomial/Transforms/MaterializeRoots.h"

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/Transforms/StaticRoots.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_MATERIALIZEROOTS
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

struct MaterializeRoots : impl::MaterializeRootsBase<MaterializeRoots> {
  void runOnOperation() override {
    auto module = getOperation();

    // First pass through and determine which rings require a root and don't
    // have one.
    llvm::SmallDenseSet<RingAttr> rings;
    auto walkResult = module->walk([&](Operation *op) {
      return llvm::TypeSwitch<Operation &, WalkResult>(*op)
          .Case<NTTOp>([&](auto op) {
            auto ring = op.getInput().getType().getRing();
            if (!ring.primitive2NthRoot()) {
              rings.insert(ring);
            }
            return WalkResult::advance();
          })
          .Case<INTTOp>([&](auto op) {
            auto ring = op.getOutput().getType().getRing();
            if (!ring.primitive2NthRoot()) {
              rings.insert(ring);
            }
            return WalkResult::advance();
          })
          .Default([&](Operation &op) { return WalkResult::advance(); });
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }

    // Second pass through and replace any rings that are in the analysis set.
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](RingAttr ring) -> std::optional<RingAttr> {
      if (rings.contains(ring)) {
        auto cmod = ring.coefficientModulus();
        auto ideal = ring.ideal();
        unsigned rootBitWidth = (cmod - 1).getActiveBits();
        auto root =
            rootBitWidth > 32
                ? roots::find64BitRoot(cmod, ideal.getDegree(), rootBitWidth)
                : roots::find32BitRoot(cmod, ideal.getDegree(), rootBitWidth);
        if (root) {
          ring = RingAttr::get(cmod, ideal, *root);
        }
      }
      return ring;
    });

    replacer.recursivelyReplaceElementsIn(module,
                                          /*replaceAttr=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
