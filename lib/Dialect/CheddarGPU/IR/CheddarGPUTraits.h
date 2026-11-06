#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_

#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::cheddar_gpu {

LogicalResult verifyLimbRangesWithinDest(Operation* op);

// Requires operand 0 to be the destination buffer; all limb range attributes
// index operand 0's limb dimension (dim 0). The operation must also constrain
// operand shapes to match the destination.
template <typename ConcreteType>
class LimbRangesWithinDest
    : public OpTrait::TraitBase<ConcreteType, LimbRangesWithinDest> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyLimbRangesWithinDest(op);
  }
};

}  // namespace mlir::heir::cheddar_gpu

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPUTRAITS_H_
