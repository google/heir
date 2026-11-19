#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_

#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::cheddar_gpu {

struct LimbRangeCheck {
  StringRef attrName;
  std::optional<ArrayRef<int64_t>> range;
  SmallVector<Value, 3> operands;
};

LogicalResult verifyLimbRange(Operation* op,
                              std::optional<ArrayRef<int64_t>> range,
                              int64_t numLimbs);

LogicalResult verifyLimbRange(Operation* op,
                              std::optional<ArrayRef<int64_t>> range,
                              Value operand, StringRef attrName = "");

LogicalResult verifyNumAux(Operation* op, std::optional<int64_t> numAux,
                           int64_t numLimbs);

template <typename ConcreteType>
class LimbRangesWithin
    : public OpTrait::TraitBase<ConcreteType, LimbRangesWithin> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    for (const LimbRangeCheck& c :
         ::llvm::cast<ConcreteType>(op).getLimbRangeChecks()) {
      for (Value v : c.operands) {
        if (failed(verifyLimbRange(op, c.range, v, c.attrName)))
          return failure();
      }
    }
    return success();
  }
};

template <typename ConcreteType>
class NumAuxWithinOut
    : public OpTrait::TraitBase<ConcreteType, NumAuxWithinOut> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    if (auto numAuxAttr = op->getAttrOfType<IntegerAttr>("num_aux")) {
      auto outType = ::llvm::dyn_cast<MemRefType>(op->getOperand(0).getType());
      if (outType && outType.getRank() >= 1) {
        return verifyNumAux(op, numAuxAttr.getInt(), outType.getDimSize(0));
      }
    }
    return success();
  }
};

LogicalResult verifyInOutCountsMatch(Operation* op);

// Requires operand 0 to be the destination buffer and operand 1 to be the
// source buffer 'in'. Verifies that the active limb count of 'in' matches the
// active limb count of 'out' (accounting for in_limb_range and out_limb_range).
template <typename ConcreteType>
class InOutCountsMatch
    : public OpTrait::TraitBase<ConcreteType, InOutCountsMatch> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyInOutCountsMatch(op);
  }
};

LogicalResult verifyNTTPrimeCounts(Operation* op);

// Requires operand 0 to be the destination buffer and operand 1 to be the
// source buffer 'in'. Verifies that prime_range count and prime_indices size
// match the active limb count of the operation.
template <typename ConcreteType>
class NTTPrimeCounts : public OpTrait::TraitBase<ConcreteType, NTTPrimeCounts> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyNTTPrimeCounts(op);
  }
};
LogicalResult verifyPadRangeWithinDest(Operation* op);

template <typename ConcreteType>
class PadRangeWithinDest
    : public OpTrait::TraitBase<ConcreteType, PadRangeWithinDest> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyPadRangeWithinDest(op);
  }
};
}  // namespace mlir::heir::cheddar_gpu

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
