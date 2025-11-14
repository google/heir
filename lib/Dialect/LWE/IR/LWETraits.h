#ifndef LIB_DIALECT_LWE_IR_LWETRAITS_H_
#define LIB_DIALECT_LWE_IR_LWETRAITS_H_

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::lwe {

// Trait that ensures that all operands and results ciphertext have the same set
// of rings.
template <typename ConcreteType>
class SameOperandsAndResultRings
    : public OpTrait::TraitBase<ConcreteType, SameOperandsAndResultRings> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    polynomial::RingAttr rings = nullptr;
    auto initOrCheckRings =
        [&](polynomial::RingAttr ring) {
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
      auto ct = dyn_cast<lwe::LWECiphertextType>(rTy);
      if (!ct) continue;
      if (failed(initOrCheckRings(ct.getCiphertextSpace().getRing()))) {
        return failure();
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<lwe::LWECiphertextType>(oTy);
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
  static LogicalResult verifyTrait(Operation* op) {
    lwe::LWEPlaintextType plaintextTypes = nullptr;
    auto initOrCheckPlaintextTypes = [&](LWEPlaintextType ps) {
      if (plaintextTypes == nullptr) {
        plaintextTypes = ps;
        return success();
      }
      if (plaintextTypes != ps) {
        op->emitOpError() << "requires all operands and results to have "
                             "the same plaintextTypes, but found "
                          << plaintextTypes << " and " << ps;
        return failure();
      }
      return success();
    };
    auto getPlaintextTypeFromCiphertextType = [&](LWECiphertextType ct) {
      return lwe::LWEPlaintextType::get(
          op->getContext(), ct.getApplicationData(), ct.getPlaintextSpace());
    };

    for (auto rTy : op->getResultTypes()) {
      auto ct = dyn_cast<lwe::LWECiphertextType>(rTy);
      if (!ct) continue;
      if (failed(initOrCheckPlaintextTypes(
              getPlaintextTypeFromCiphertextType(ct)))) {
        return failure();
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<lwe::LWECiphertextType>(oTy);
      auto pt = dyn_cast<lwe::LWEPlaintextType>(oTy);
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

// Trait that ensures that all ciphertext types match.
template <typename ConcreteType>
class AllCiphertextTypesMatch
    : public OpTrait::TraitBase<ConcreteType, AllCiphertextTypesMatch> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    LWECiphertextType ciphertextTypes = nullptr;
    auto initOrCheckCiphertextTypes = [&](LWECiphertextType ct) {
      if (ciphertextTypes == nullptr) {
        ciphertextTypes = ct;
        return success();
      }
      if (ciphertextTypes != ct) {
        op->emitOpError() << "requires all ciphertexts to have "
                             "the same ciphertextType";
        return failure();
      }
      return success();
    };

    for (auto rTy : op->getResultTypes()) {
      auto ct = dyn_cast<lwe::LWECiphertextType>(rTy);
      if (!ct) continue;
      if (failed(initOrCheckCiphertextTypes(ct))) {
        return failure();
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      auto ct = dyn_cast<lwe::LWECiphertextType>(oTy);
      if (!ct) continue;  // Check only ciphertexts
      if (ct && failed(initOrCheckCiphertextTypes(ct))) {
        return failure();
      }
    }
    return success();
  }
};

// Helper that verifies if an op is a ciphertext plaintext operation.
inline LogicalResult verifyCiphertextPlaintextOp(Operation* op) {
  if (op->getNumOperands() != 2) {
    return op->emitOpError()
           << "ciphertext plaintext operation requires two operands";
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError()
           << "ciphertext plaintext operation requires one result";
  }

  auto operandTys = op->getOperandTypes();
  if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(operandTys[0]))) {
    if (!isa<lwe::LWEPlaintextType>(getElementTypeOrSelf(operandTys[1]))) {
      return op->emitOpError()
             << "expected ciphertext, plaintext operand types, got "
             << operandTys[0] << ", " << operandTys[1];
    }
  } else if (isa<lwe::LWEPlaintextType>(getElementTypeOrSelf(operandTys[0]))) {
    if (!isa<lwe::LWECiphertextType>(getElementTypeOrSelf(operandTys[1]))) {
      return op->emitOpError()
             << "expected plaintext, ciphertext operand types, got "
             << operandTys[0] << ", " << operandTys[1];
    }
  } else {
    return op->emitOpError()
           << "expected first operand to be a (tensor of) ciphertext or "
              "plaintext type, got "
           << operandTys[0];
  }
  if (!isa<lwe::LWECiphertextType>(
          getElementTypeOrSelf(op->getResultTypes()[0]))) {
    return op->emitError()
           << "expected result to be (tensor of) ciphertext, got "
           << op->getResultTypes()[0];
  }
  return success();
}

// Trait that verifies a binary operation between ciphertext and plaintext in
// any order.
template <typename ConcreteType>
class IsCiphertextPlaintextOp
    : public OpTrait::TraitBase<ConcreteType, IsCiphertextPlaintextOp> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyCiphertextPlaintextOp(op);
  }
};

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_IR_LWETRAITS_H_
