#include "lib/Dialect/CheddarGPU/IR/CheddarGPUTraits.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::cheddar_gpu {

LogicalResult verifyLimbRange(Operation* op,
                              std::optional<ArrayRef<int64_t>> range,
                              int64_t numLimbs) {
  if (!range) return success();
  // Shape and signs are enforced by CheddarGPU_LimbRangeAttr.
  int64_t offset = (*range)[0];
  int64_t count = (*range)[1];
  if (count > numLimbs - offset) {
    return op->emitOpError()
           << "limb range [" << offset << ", " << count
           << "] out of bounds for memref with " << numLimbs << " limbs";
  }

  if (auto mod = op->getParentOfType<ModuleOp>()) {
    if (auto params = mod->getAttrOfType<ParamsAttr>(
            CheddarGPUDialect::kCheddarGpuParamsAttr)) {
      int64_t paramsLimbs = static_cast<int64_t>(params.getNumLimbs());
      if (count > paramsLimbs - offset) {
        return op->emitOpError()
               << "limb range [" << offset << ", " << count
               << "] exceeds module params limb count (" << paramsLimbs << ")";
      }
    }
  }
  return success();
}

LogicalResult verifyLimbRange(Operation* op,
                              std::optional<ArrayRef<int64_t>> range,
                              Value operand, StringRef /*attrName*/) {
  if (!range) return success();
  auto memrefType = ::llvm::dyn_cast<MemRefType>(operand.getType());
  if (!memrefType || memrefType.getRank() < 1) return success();
  return verifyLimbRange(op, range, memrefType.getDimSize(0));
}

LogicalResult verifyNumAux(Operation* op, std::optional<int64_t> numAux,
                           int64_t numLimbs) {
  if (numAux) {
    // num_aux >= 0 is enforced by CheddarGPU_NonNegI64Attr.
    if (*numAux > numLimbs) {
      return op->emitOpError("num_aux must be in [0, ")
             << numLimbs << "], but got " << *numAux;
    }
  }
  return success();
}

LogicalResult verifyInOutCountsMatch(Operation* op) {
  if (op->getNumOperands() < 2) return success();
  auto outType = ::llvm::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto inType = ::llvm::dyn_cast<ShapedType>(op->getOperand(1).getType());
  if (!outType || !inType || outType.getRank() < 1 || inType.getRank() < 1)
    return success();

  auto getLimbCount = [&](StringRef attrName, ShapedType type) {
    if (auto range = op->getAttrOfType<DenseI64ArrayAttr>(attrName)) {
      assert(range.size() == 2 && "guaranteed by CheddarGPU_LimbRangeAttr");
      return range.asArrayRef()[1];
    }
    return type.getDimSize(0);
  };

  int64_t outCount = getLimbCount("out_limb_range", outType);
  int64_t inCount = getLimbCount("in_limb_range", inType);

  if (inCount != outCount) {
    return op->emitOpError("limb count mismatch between in and out (")
           << inCount << " vs " << outCount << ")";
  }
  return success();
}

LogicalResult verifyNTTPrimeCounts(Operation* op) {
  if (op->getNumOperands() < 2) return success();
  auto inType = ::llvm::dyn_cast<MemRefType>(op->getOperand(1).getType());
  if (!inType || inType.getRank() < 1) return success();
  int64_t numLimbs = inType.getDimSize(0);

  auto outRange = op->getAttrOfType<DenseI64ArrayAttr>("out_limb_range");
  auto inRange = op->getAttrOfType<DenseI64ArrayAttr>("in_limb_range");
  if (outRange && inRange &&
      outRange.asArrayRef()[1] != inRange.asArrayRef()[1]) {
    return op->emitOpError(
        "limb count mismatch between in_limb_range and out_limb_range");
  }

  int64_t limbCount = inRange ? inRange.asArrayRef()[1] : numLimbs;

  if (auto primeRange = op->getAttrOfType<DenseI64ArrayAttr>("prime_range")) {
    int64_t count = primeRange.asArrayRef()[1];
    if (count != limbCount) {
      return op->emitOpError(
          "prime_range count mismatch with limb range count");
    }
  }

  if (auto primeIndices =
          op->getAttrOfType<DenseI64ArrayAttr>("prime_indices")) {
    if (static_cast<int64_t>(primeIndices.asArrayRef().size()) != limbCount) {
      return op->emitOpError(
          "prime_indices count mismatch with limb range count");
    }
  }

  return success();
}
LogicalResult verifyPadRangeWithinDest(Operation* op) {
  if (op->getNumOperands() == 0) return success();
  auto outType = ::llvm::dyn_cast<MemRefType>(op->getOperand(0).getType());
  if (!outType || outType.getRank() < 1) return success();
  auto padStartAttr = op->getAttrOfType<IntegerAttr>("pad_start");
  auto padEndAttr = op->getAttrOfType<IntegerAttr>("pad_end");
  if (!padStartAttr || !padEndAttr) return success();
  int64_t padStart = padStartAttr.getInt();
  int64_t padEnd = padEndAttr.getInt();
  int64_t numLimbs = outType.getDimSize(0);
  if (padStart < 0 || padEnd < padStart || padEnd > numLimbs) {
    return op->emitOpError("pad range [")
           << padStart << ", " << padEnd << "] out of bounds for output with "
           << numLimbs << " limbs";
  }
  return success();
}
}  // namespace mlir::heir::cheddar_gpu
