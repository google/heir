#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"     // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {

FailureOr<uint64_t> getConstantTripCount(Operation* op) {
  if (auto loopLikeOp = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<APInt> tripCount = loopLikeOp.getStaticTripCount();
    if (tripCount.has_value()) return tripCount.value().getZExtValue();
  }

  // Fall back if LoopLikeOpInterface fails, and at the moment affine.for
  // doesn't implement LoopLikeOpInterface.
  if (auto affineFor = dyn_cast<affine::AffineForOp>(op)) {
    std::optional<uint64_t> maybeTripCount =
        affine::getConstantTripCount(affineFor);
    if (maybeTripCount.has_value()) return *maybeTripCount;
  }

  return failure();
}

FailureOr<int64_t> calculateSize(Operation* op) {
  int64_t size = 1;
  Operation* parent = op->getParentOp();
  while (parent) {
    if (isa<affine::AffineForOp, LoopLikeOpInterface>(parent)) {
      FailureOr<uint64_t> tripCount = getConstantTripCount(parent);
      // If we can't detect a static trip count, something went terribly wrong
      // elsewhere.
      if (failed(tripCount)) {
        return parent->emitError()
               << "Could not resolve constant trip count for loop";
      }
      // Exit early if the product of loop iterations overflows int64.
      if (__builtin_mul_overflow(size, *tripCount, &size)) {
        return parent->emitError()
               << "64-bit integer overflow during size calculation";
      }
    }
    parent = parent->getParentOp();
  }
  return size;
}

}  // namespace

PreprocessingStorageLayoutAnalysis::PreprocessingStorageLayoutAnalysis(
    Operation* op) {
  DenseMap<uint32_t, preprocessing::StoreOp> storesBySite;

  WalkResult storeResult = op->walk([&](preprocessing::StoreOp storeOp) {
    uint32_t siteId = storeOp.getSiteId();
    if (storesBySite.count(siteId)) {
      storeOp->emitError() << "Duplicate StoreOp found for site_id: " << siteId;
      return WalkResult::interrupt();
    }
    storesBySite[siteId] = storeOp;
    return WalkResult::advance();
  });

  if (storeResult.wasInterrupted()) {
    valid = false;
    return;
  }

  WalkResult loadResult =
      op->walk([&](preprocessing::LoadOp loadOp) {
        uint32_t siteId = loadOp.getSiteId();
        if (!storesBySite.count(siteId)) {
          loadOp->emitError() << "LoadOp found with site_id " << siteId
                              << " but no corresponding StoreOp exists";
          return WalkResult::interrupt();
        }
        auto storeOp = storesBySite[siteId];
        if (loadOp.getElementType() != storeOp.getElementType()) {
          loadOp->emitError()
              << "LoadOp element type " << loadOp.getElementType()
              << " does not match corresponding StoreOp element type "
              << storeOp.getElementType();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

  if (loadResult.wasInterrupted()) {
    valid = false;
    return;
  }

  SmallVector<uint32_t, 4> siteIds;
  for (const auto& pair : storesBySite) {
    siteIds.push_back(pair.first);
  }
  llvm::sort(siteIds);

  DenseMap<Type, int64_t> currentOffsets;
  for (uint32_t siteId : siteIds) {
    preprocessing::StoreOp storeOp = storesBySite[siteId];
    Type elementType = storeOp.getElementType();
    FailureOr<int64_t> size = calculateSize(storeOp);
    if (failed(size)) {
      valid = false;
      return;
    }
    int64_t currentOffset = currentOffsets[elementType];
    siteLayouts[elementType][siteId] = {currentOffset, *size};
    if (__builtin_add_overflow(currentOffset, *size, &currentOffset)) {
      storeOp->emitError()
          << "64-bit integer overflow during offset accumulation";
      valid = false;
      return;
    }
    currentOffsets[elementType] = currentOffset;
  }
  totalSizes = std::move(currentOffsets);
  valid = true;
}

FailureOr<SiteLayout> PreprocessingStorageLayoutAnalysis::getLayout(
    Type type, uint32_t siteId) const {
  auto typeIt = siteLayouts.find(type);
  if (typeIt == siteLayouts.end()) return failure();
  const auto& innerMap = typeIt->second;
  auto siteIt = innerMap.find(siteId);
  if (siteIt == innerMap.end()) return failure();
  return siteIt->second;
}

FailureOr<int64_t> PreprocessingStorageLayoutAnalysis::getTotalSize(
    Type type) const {
  auto it = totalSizes.find(type);
  if (it == totalSizes.end()) return failure();
  return it->second;
}

}  // namespace heir
}  // namespace mlir
