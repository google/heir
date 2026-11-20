#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"

#include <cstddef>
#include <cstdint>

#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
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

// === HoistBSFusedOp ===

LogicalResult HoistBSFusedOp::verify() {
  size_t R = getOutBx().size();
  size_t beta = getModUp().size();
  auto firstOutBxType = ::llvm::cast<MemRefType>(getOutBx()[0].getType());
  Type elemType = firstOutBxType.getElementType();

  auto slabType = ::llvm::cast<MemRefType>(getEvkSlab().getType());

  ArrayRef<int64_t> evkOffsets = getEvkByteOffsets();
  if (static_cast<size_t>(evkOffsets.size()) != R &&
      static_cast<size_t>(evkOffsets.size()) != R * beta * 2) {
    return emitOpError("evk_byte_offsets length (")
           << evkOffsets.size() << ") must be either R (" << R
           << ") or R*beta*2 (" << R * beta * 2 << ")";
  }
  int64_t wordBytes = elemType.getIntOrFloatBitWidth() / 8;
  for (size_t i = 0; i < evkOffsets.size(); ++i) {
    int64_t off = evkOffsets[i];
    if (off < 0 || off % wordBytes != 0) {
      return emitOpError("evk_byte_offsets[")
             << i << "] (" << off << ") must be non-negative and a multiple of "
             << wordBytes;
    }
    if (off >= slabType.getDimSize(0)) {
      return emitOpError("evk_byte_offsets[")
             << i << "] (" << off << ") exceeds evk_slab size ("
             << slabType.getDimSize(0) << ")";
    }
  }
  return success();
}

// === HoistGSFusedOp ===

LogicalResult HoistGSFusedOp::verify() {
  size_t G = getOutBx().size();
  size_t B = getInBx().size();
  auto firstOutBxType = ::llvm::cast<MemRefType>(getOutBx()[0].getType());
  int64_t numLimbs = firstOutBxType.getDimSize(0);
  int64_t ptLimbs = getPtLimbs();
  if (ptLimbs < numLimbs) {
    return emitOpError("pt_limbs (")
           << ptLimbs << ") must be >= ciphertext limbs (" << numLimbs << ")";
  }

  ArrayRef<int64_t> ptOffsets = getPtWordOffsets();
  if (static_cast<size_t>(ptOffsets.size()) != G * B) {
    return emitOpError("pt_word_offsets length (")
           << ptOffsets.size() << ") must equal G * B (" << G * B << ")";
  }
  int64_t wordBytes =
      firstOutBxType.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t slabBytes =
      ::llvm::cast<MemRefType>(getConstsSlab().getType()).getDimSize(0);
  for (size_t i = 0; i < ptOffsets.size(); ++i) {
    int64_t off = ptOffsets[i];
    if (off >= 0 && (slabBytes <= 0 || off > (slabBytes - 1) / wordBytes)) {
      return emitOpError("pt_word_offsets[")
             << i << "] (" << off << ") exceeds consts_slab size (" << slabBytes
             << " bytes)";
    }
  }
  return success();
}

// === HoistPhaseOp ===

LogicalResult HoistPhaseOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());

  HoistPhaseKind kind = getKind();
  int64_t phase = getPhase();
  int64_t expectedInLimbs = 0;
  int64_t expectedOutLimbs = 0;
  switch (kind) {
    case HoistPhaseKind::CTS:
      if (phase < 0 || phase > 3) {
        return emitOpError("phase for 'cts' must be in [0, 3], but got ")
               << phase;
      }
      expectedInLimbs = 26 - phase;
      expectedOutLimbs = 25 - phase;
      break;
    case HoistPhaseKind::STC:
      if (phase < 0 || phase > 2) {
        return emitOpError("phase for 'stc' must be in [0, 2], but got ")
               << phase;
      }
      expectedInLimbs = 14 - phase;
      expectedOutLimbs = 13 - phase;
      break;
  }

  int64_t inLimbs = inBxType.getDimSize(0);
  int64_t outLimbs = outBxType.getDimSize(0);
  if (inLimbs != expectedInLimbs) {
    return emitOpError("input limb count for ")
           << stringifyHoistPhaseKind(kind) << " phase " << phase << " must be "
           << expectedInLimbs << ", but got " << inLimbs;
  }
  if (outLimbs != expectedOutLimbs) {
    return emitOpError("output limb count for ")
           << stringifyHoistPhaseKind(kind) << " phase " << phase << " must be "
           << expectedOutLimbs << ", but got " << outLimbs;
  }

  return success();
}

}  // namespace mlir::heir::cheddar_gpu
