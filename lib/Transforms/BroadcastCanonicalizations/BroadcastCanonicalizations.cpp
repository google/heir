#include "lib/Transforms/BroadcastCanonicalizations/BroadcastCanonicalizations.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <list>
#include <optional>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "broadcast-canonicalizations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_BROADCASTCANONICALIZATIONS
#include "lib/Transforms/BroadcastCanonicalizations/BroadcastCanonicalizations.h.inc"

struct GenericToBroadcast : public OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  GenericToBroadcast(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::GenericOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[generic-to-broadcast] Trying to rewrite: ");
    LLVM_DEBUG(genericOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // make sure we are using parallel iterators
    for (int i = 0; i < genericOp.getIteratorTypesArray().size(); i++) {
      auto itType = genericOp.getIteratorTypesArray()[i];
      if (itType != utils::IteratorType::parallel) {
        LLVM_DEBUG(llvm::dbgs() << "[generic-to-broadcast] Iterator type " << i
                                << " is not parallel\n");
        return rewriter.notifyMatchFailure(
            genericOp,
            "genericOp with non-parallel iterator type not "
            "supported");
      }
    }

    int64_t numInputDims =
        llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[0])
            .getAffineMap()
            .getNumDims();

    // check that the op has 1 input and 1 initial value
    if (genericOp.getInputs().size() != 1 ||
        genericOp.getDpsInits().size() != 1 ||
        genericOp.getIndexingMaps().size() != 2) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[generic-to-broadcast] Number of inputs: "
                 << genericOp.getInputs().size()
                 << " Number of outputs: " << genericOp.getDpsInits().size()
                 << " Number of indexing maps: "
                 << genericOp.getIndexingMaps().size() << "\n");
      return rewriter.notifyMatchFailure(
          genericOp,
          "genericOp with more than 1 input or output not supported");
    }

    // check all affine maps
    mlir::SmallVector<int64_t> dimensions = {};  // broadcast dimensions
    for (int i = 0; i < genericOp.getIndexingMaps().size(); i++) {
      // check that they all have the same number of input dimensions
      if (llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
              .getAffineMap()
              .getNumDims() != numInputDims) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[generic-to-broadcast] Map " << i
                   << " does not have the same number of input dimensions\n");
        return rewriter.notifyMatchFailure(genericOp,
                                           "genericOp with incompatible number "
                                           "of input dimensions not supported");
      }
      // check that the genericOp has a linalg.yield as the only op in the block

      if (genericOp.getRegion().getBlocks().size() != 1 ||
          genericOp.getRegion().getBlocks().front().getOperations().size() !=
              1 ||
          llvm::isa_and_nonnull<linalg::YieldOp>(
              genericOp.getRegion().getBlocks().front().getTerminator()) ==
              false) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[generic-to-broadcast] genericOp does not have exactly "
                      "1 block with a linalg.yield terminator\n");
        return rewriter.notifyMatchFailure(
            genericOp, "genericOp with unsupported region not supported");
      }

      // check maps
      if (i < genericOp.getInputs().size()) {  // input maps
        if (!checkInputMap(
                llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
                    .getAffineMap(),
                dimensions)) {
          LLVM_DEBUG(llvm::dbgs() << "[generic-to-broadcast] Input map " << i
                                  << " is not right\n");
          return rewriter.notifyMatchFailure(
              genericOp,
              "genericOp with incompatible output dimensions not supported");
        } else if (i >= genericOp.getInputs().size()) {  // output maps
          if (!llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
                   .getAffineMap()
                   .isIdentity()) {
            LLVM_DEBUG(llvm::dbgs() << "[generic-to-broadcast] Output map " << i
                                    << " is not identity\n");
            return rewriter.notifyMatchFailure(
                genericOp,
                "genericOp with incompatible output map not supported");
          }

          LLVM_DEBUG(llvm::dbgs() << "[generic-to-broadcast] Output map " << i
                                  << " is right\n");
        }
      }
    }
    ImplicitLocOpBuilder builder(genericOp.getLoc(), rewriter);

    // make a linalg.broadcastOp to replace the linalg.genericOp
    mlir::linalg::BroadcastOp broadcast = mlir::linalg::BroadcastOp::create(
        builder, genericOp.getInputs()[0], genericOp.getDpsInits()[0],
        dimensions);
    // don't need to clone the block

    rewriter.replaceAllOpUsesWith(genericOp, broadcast.getResults());
    rewriter.eraseOp(genericOp);

    return success();
  }

 private:
  // This function assures that the input map is an identity map with the
  // broadcast dimensions removed. Ex: (d0, d1) -> (d0) or (d0) -> ().
  bool checkInputMap(const AffineMap map,
                     mlir::SmallVector<int64_t>& dimensions) const {
    // check total size
    LLVM_DEBUG(llvm::dbgs() << " # of map dims: " << map.getNumDims()
                            << " # of results: " << map.getNumResults());
    if (map.getNumResults() == map.getNumDims()) {
      LLVM_DEBUG(llvm::dbgs() << "does not have broadcast dimensions");
      return false;
    }
    // Example pass: (d0) -> ()
    if (map.getNumResults() == 0) {
      // add all dimensions as broadcast dimensions
      for (unsigned i = 0, numDims = map.getNumDims(); i < numDims; ++i) {
        dimensions.push_back(i);
      }
      return true;
    }

    llvm::SmallVector<int64_t> seenDims = {};
    for (unsigned i = 0, numResults = map.getNumResults(); i < numResults;
         ++i) {
      auto expr = dyn_cast<AffineDimExpr>(map.getResult(i));
      if (!expr) {
        LLVM_DEBUG(llvm::dbgs()
                   << "result " << i << " is not an affine dim expr\n");
        return false;
      }
      int64_t dimPosition = expr.getPosition();
      if (dimPosition >= map.getNumDims()) {
        LLVM_DEBUG(llvm::dbgs() << "result " << i << " has dim position "
                                << dimPosition << " which is out of bounds\n");
        return false;
      }
      if (llvm::is_contained(seenDims, dimPosition)) {
        LLVM_DEBUG(llvm::dbgs() << "result " << i << " has dim position "
                                << dimPosition << " which is already seen\n");
        return false;
      }
      seenDims.push_back(dimPosition);
    }

    // subtract the seen dimensions from the total dimensions to get the
    // broadcast dimensions
    for (unsigned i = 0, numDims = map.getNumDims(); i < numDims; ++i) {
      if (!llvm::is_contained(seenDims, i)) {
        dimensions.push_back(i);
      }
    }
    return true;
  }
};

struct BroadcastCanonicalizations
    : public impl::BroadcastCanonicalizationsBase<BroadcastCanonicalizations> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<GenericToBroadcast>(context);

    walkAndApplyPatterns(module, std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
