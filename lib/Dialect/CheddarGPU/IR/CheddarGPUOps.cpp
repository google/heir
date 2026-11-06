#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"

#include <cstdint>

#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

using ::mlir::DenseI64ArrayAttr;

#define GET_OP_CLASSES
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.cpp.inc"

namespace mlir::heir::cheddar_gpu {

// === NTTOp ===

LogicalResult NTTOp::verify() {
  if (getSkipRange() && getSkipCopySrc()) {
    int64_t skipStart = (*getSkipRange())[0];
    int64_t skipEnd = (*getSkipRange())[1];
    int64_t skipCount = skipEnd - skipStart;
    int64_t skipInOffset = getSkipCopyInOffset();
    auto skipSrcType = ::llvm::cast<MemRefType>(getSkipCopySrc().getType());
    if (skipCount > skipSrcType.getDimSize(0) - skipInOffset) {
      return emitOpError("skip_copy_src buffer too small for skipped range");
    }
  }
  return success();
}

// === INTTOp ===

LogicalResult INTTOp::verify() {
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  int64_t numLimbs = outType.getDimSize(0);
  int64_t limbCount =
      getOutLimbRange()
          ? (*getOutLimbRange())[1]
          : (getInLimbRange() ? (*getInLimbRange())[1] : numLimbs);
  int64_t inOffset = getInLimbRange() ? (*getInLimbRange())[0] : 0;

  if (getInSplit()) {
    int64_t splitCount = (*getInSplit())[0];
    int64_t extraLimbs = (*getInSplit())[1];
    if (splitCount > limbCount) {
      return emitOpError("in_split split_count must be in [0, limb_count]");
    }
    int64_t requiredExtent = inOffset + limbCount;
    if (splitCount < limbCount) {
      requiredExtent += extraLimbs;
    }
    if (inType.getDimSize(0) < requiredExtent) {
      return emitOpError("input buffer too small for in_split extent");
    }
  }

  if (getCst()) {
    auto cstType = ::llvm::cast<MemRefType>(getCst().getType());
    if (cstType.getDimSize(0) != limbCount) {
      return emitOpError("constant buffer size (")
             << cstType.getDimSize(0) << ") must match limb count ("
             << limbCount << ")";
    }
  }

  return success();
}

}  // namespace mlir::heir::cheddar_gpu
