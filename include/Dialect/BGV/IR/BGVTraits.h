#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVTRAITS_H_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVTRAITS_H_

#include "include/Dialect/BGV/IR/BGVAttributes.h"
#include "include/Dialect/BGV/IR/BGVTypes.h"
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project

namespace mlir::heir::bgv {

// Trait that ensures that all operands and results ciphertext have the same set
// of rings.
template <typename ConcreteType>
class SameOperandsAndResultRings
    : public OpTrait::TraitBase<ConcreteType, SameOperandsAndResultRings> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    BGVRingsAttr rings = nullptr;
    for (auto rTy : op->getResultTypes()) {
      auto ct = dyn_cast<CiphertextType>(rTy);
      if (!ct) continue;
      if (rings == nullptr) {
        rings = ct.getRings();
        continue;
      }
      if (rings != ct.getRings()) {
        return op->emitOpError()
               << "requires all operands and results to have the same rings";
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<CiphertextType>(oTy);
      if (!ct) continue;  // Check only ciphertexts

      if (rings == nullptr) {
        rings = ct.getRings();
        continue;
      }

      if (rings != ct.getRings()) {
        return op->emitOpError()
               << "requires all operands and results to have the same rings";
      }
    }
    return success();
  }
};

}  // namespace mlir::heir::bgv

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVTRAITS_H_
