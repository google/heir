#ifndef LIB_DIALECT_LWE_IR_LWETRAITS_H_
#define LIB_DIALECT_LWE_IR_LWETRAITS_H_

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
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
    ::mlir::heir::polynomial::RingAttr rings = nullptr;
    auto initOrCheckRings =
        [&](::mlir::heir::polynomial::RingAttr ring) {
          if (rings == nullptr) {
            rings = ring;
            return success();
          }
          if (rings != ring) {
            op->emitOpError()
                << "requires all operands and results to have the same rings";
            return failure();
          }
          return success();
        };
    for (auto rTy : op->getResultTypes()) {
      auto ct = dyn_cast<lwe::NewLWECiphertextType>(rTy);
      if (!ct) continue;
      if (failed(initOrCheckRings(ct.getCiphertextSpace().getRing()))) {
        return failure();
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<lwe::NewLWECiphertextType>(oTy);
      if (!ct) continue;  // Check only ciphertexts
      if (failed(initOrCheckRings(ct.getCiphertextSpace().getRing()))) {
        return failure();
      }
    }
    return success();
  }
};

// Trait that ensures that all operands and results ciphertext/plaintext have
// the same set of application space/plaintext spaces.
template <typename ConcreteType>
class SameOperandsAndResultPlaintextTypes
    : public OpTrait::TraitBase<ConcreteType,
                                SameOperandsAndResultPlaintextTypes> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    lwe::NewLWEPlaintextType plaintextTypes = nullptr;
    auto initOrCheckPlaintextTypes = [&](NewLWEPlaintextType ps) {
      if (plaintextTypes == nullptr) {
        plaintextTypes = ps;
        return success();
      }
      if (plaintextTypes != ps) {
        op->emitOpError() << "requires all operands and results to have "
                             "the same plaintextTypes";
        return failure();
      }
      return success();
    };
    auto getPlaintextTypeFromCiphertextType = [&](NewLWECiphertextType ct) {
      return lwe::NewLWEPlaintextType::get(
          op->getContext(), ct.getApplicationData(), ct.getPlaintextSpace());
    };

    for (auto rTy : op->getResultTypes()) {
      auto ct = dyn_cast<lwe::NewLWECiphertextType>(rTy);
      if (!ct) continue;
      if (failed(initOrCheckPlaintextTypes(
              getPlaintextTypeFromCiphertextType(ct)))) {
        return failure();
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<lwe::NewLWECiphertextType>(oTy);
      auto pt = dyn_cast<lwe::NewLWEPlaintextType>(oTy);
      if (!ct && !pt) continue;  // Check only ciphertexts and plaintexts
      if (ct && failed(initOrCheckPlaintextTypes(
                    getPlaintextTypeFromCiphertextType(ct)))) {
        return failure();
      }
      if (pt && failed(initOrCheckPlaintextTypes(pt))) {
        return failure();
      }
    }
    return success();
  }
};

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_IR_LWETRAITS_H_
