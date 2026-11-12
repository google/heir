#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_

#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::cheddar_gpu {

LogicalResult verifyLimbRangesWithin(Operation* op);

// Verifies that operand-specific limb ranges (out_limb_range, in_limb_range,
// lhs_limb_range, rhs_limb_range) are within bounds of their corresponding
// operand's limb dimension (dim 0). Also verifies num_aux against operand 0.
template <typename ConcreteType>
class LimbRangesWithin
    : public OpTrait::TraitBase<ConcreteType, LimbRangesWithin> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyLimbRangesWithin(op);
  }
};

template <typename ConcreteType>
using LimbRangesWithinDest = LimbRangesWithin<ConcreteType>;

inline LogicalResult verifyLimbRangesWithinDest(Operation* op) {
  return verifyLimbRangesWithin(op);
}

template <typename ConcreteType>
using InOutLimbRangesWithin = LimbRangesWithin<ConcreteType>;

inline LogicalResult verifyInOutLimbRangesWithin(Operation* op) {
  return verifyLimbRangesWithin(op);
}

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
}  // namespace mlir::heir::cheddar_gpu

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
