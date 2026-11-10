#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_

#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
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

}  // namespace mlir::heir::cheddar_gpu

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
