#ifndef HEIR_LIB_DIALECT_LWE_IR_LWETRAITS_H_
#define HEIR_LIB_DIALECT_LWE_IR_LWETRAITS_H_

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::lwe {

// Trait that ensures that all operands and results ciphertext have the same set
// of rings.
template <typename ConcreteType>
class SameOperandsAndResultRings
    : public OpTrait::TraitBase<ConcreteType, SameOperandsAndResultRings> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    ::mlir::polynomial::RingAttr rings = nullptr;
    for (auto rTy : op->getResultTypes()) {
      auto ct = dyn_cast<lwe::RLWECiphertextType>(rTy);
      if (!ct) continue;
      if (rings == nullptr) {
        rings = ct.getRlweParams().getRing();
        continue;
      }
      if (rings != ct.getRlweParams().getRing()) {
        return op->emitOpError()
               << "requires all operands and results to have the same rings";
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<lwe::RLWECiphertextType>(oTy);
      if (!ct) continue;  // Check only ciphertexts

      if (rings == nullptr) {
        rings = ct.getRlweParams().getRing();
        continue;
      }

      if (rings != ct.getRlweParams().getRing()) {
        return op->emitOpError()
               << "requires all operands and results to have the same rings";
      }
    }
    return success();
  }
};

}  // namespace mlir::heir::lwe

#endif  // HEIR_LIB_DIALECT_LWE_IR_LWETRAITS_H_
