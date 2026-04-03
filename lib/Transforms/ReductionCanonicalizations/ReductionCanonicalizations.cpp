#include "lib/Transforms/ReductionCanonicalizations/ReductionCanonicalizations.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineMap.h>
#include <sys/socket.h>

#include <algorithm>
#include <cassert>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"          // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "reduction-canonicalizations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_REDUCTIONCANONICALIZATIONS
#include "lib/Transforms/ReductionCanonicalizations/ReductionCanonicalizations.h.inc"

struct GenericToReduce : public OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  GenericToReduce(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::GenericOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[generic-to-reduce] Trying to rewrite: ");
    LLVM_DEBUG(genericOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // make sure we need to reduce and collect dimensions
    mlir::SmallVector<int64_t> dimensions = {};
    for (int i = 0; i < genericOp.getIteratorTypesArray().size(); i++) {
      auto itType = genericOp.getIteratorTypesArray()[i];
      if (itType == utils::IteratorType::reduction) {
        dimensions.push_back(i);
      } else if (itType != utils::IteratorType::parallel) {
        return rewriter.notifyMatchFailure(
            genericOp,
            "genericOp with non-reduction, non-parallel iterator type not "
            "supported");
      }
    }

    if (dimensions.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[generic-to-reduce] No reduction dimensions found\n");
      return rewriter.notifyMatchFailure(
          genericOp, "genericOp with no reduction dimensions not supported");
    }

    // check all affine maps
    for (int i = 0; i < genericOp.getIndexingMaps().size(); i++) {
      if (i < genericOp.getInputs().size()) {  // input maps
        if (!llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
                 .getAffineMap()
                 .isIdentity()) {
          return rewriter.notifyMatchFailure(
              genericOp,
              "genericOp with incompatible input dimensions not supported");
        }
      } else {  // output maps
        if (!checkOutputMap(
                llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
                    .getAffineMap(),
                dimensions)) {
          LLVM_DEBUG(llvm::dbgs() << "[generic-to-reduce] Output map " << i
                                  << " is not right\n");
          return rewriter.notifyMatchFailure(
              genericOp,
              "genericOp with incompatible output dimensions not supported");
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "[generic-to-reduce] Output map " << i << " is right\n");
      }
    }

    // make a linalg.reduceOp to replace the linalg.genericOp
    mlir::linalg::ReduceOp reduce = mlir::linalg::ReduceOp::create(
        rewriter, genericOp.getLoc(), TypeRange(genericOp.getDpsInits()),
        genericOp.getInputs(), SmallVector<Value>(genericOp.getDpsInits()),
        dimensions);

    // generic ops have 1 block
    rewriter.cloneRegionBefore(genericOp.getRegion(), reduce.getRegion(),
                               reduce.getRegion().end());

    rewriter.replaceAllOpUsesWith(genericOp, reduce.getResults());
    rewriter.eraseOp(genericOp);

    return success();
  }

 private:
  // This function assures that the output map is an identity map with the
  // reduction dimensions removed. Ex: (d0, d1) -> (d0) or (d0) -> ().
  bool checkOutputMap(const AffineMap map,
                      const mlir::SmallVector<int64_t> dimensions) const {
    // check total size
    LLVM_DEBUG(llvm::dbgs() << "# of reduction dims: " << dimensions.size()
                            << " # of map dims: " << map.getNumDims()
                            << " # of results: " << map.getNumResults());
    if (map.getNumResults() != map.getNumDims() - dimensions.size()) {
      LLVM_DEBUG(llvm::dbgs() << "not right # of map dims");
      return false;
    }
    // Example pass: (d0) -> ()
    if (map.getNumResults() == 0) {
      return true;
    }
    // (d0, d1) -> (d0)
    int resultIndex =
        0;  // output map index, i is just the index in the map inputs
    ArrayRef<AffineExpr> results = map.getResults();
    for (unsigned i = 0, numDims = map.getNumResults(); i < numDims; ++i) {
      bool isReduced = find(dimensions, i) != dimensions.end();
      if (isReduced) {  // skip this on the output dimension
        continue;
      }
      // based on isIdentity
      auto expr = dyn_cast<AffineDimExpr>(results[resultIndex]);
      if (!expr || expr.getPosition() != i) {
        LLVM_DEBUG(llvm::dbgs()
                   << "map result " << expr.getPosition()
                   << " is not an affine dim expr with expected position "
                   << resultIndex);
        return false;
      }
      resultIndex++;
    }
    return true;
  }
};

struct ReductionCanonicalizations
    : public impl::ReductionCanonicalizationsBase<ReductionCanonicalizations> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<GenericToReduce>(context);

    walkAndApplyPatterns(module, std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
