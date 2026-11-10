#include "lib/Dialect/CheddarGPU/IR/CheddarGPUTraits.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
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

namespace {

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
    if (auto params = mod->getAttrOfType<ParamsAttr>("cheddar_gpu.params")) {
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

}  // namespace

LogicalResult verifyLimbRangesWithin(Operation* op) {
  if (op->getNumOperands() == 0) return success();

  auto checkRange = [&](unsigned opIdx, StringRef attrName) -> LogicalResult {
    if (opIdx >= op->getNumOperands()) return success();
    auto type = ::llvm::dyn_cast<MemRefType>(op->getOperand(opIdx).getType());
    if (!type || type.getRank() < 1) return success();
    if (auto range = op->getAttrOfType<DenseI64ArrayAttr>(attrName))
      return verifyLimbRange(op, range.asArrayRef(), type.getDimSize(0));
    return success();
  };

  if (failed(checkRange(0, "out_limb_range"))) return failure();
  if (failed(checkRange(1, "lhs_limb_range"))) return failure();
  if (failed(checkRange(2, "rhs_limb_range"))) return failure();

  if (auto numAuxAttr = op->getAttrOfType<IntegerAttr>("num_aux")) {
    auto outType = ::llvm::dyn_cast<MemRefType>(op->getOperand(0).getType());
    if (outType && outType.getRank() >= 1) {
      if (failed(verifyNumAux(op, numAuxAttr.getInt(), outType.getDimSize(0))))
        return failure();
    }
  }

  return success();
}

}  // namespace mlir::heir::cheddar_gpu
