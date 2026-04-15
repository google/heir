#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTRAITS_H_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTRAITS_H_

#include <optional>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::polynomial {

// Helper to verify that all polynomial operands and results have a specific
// form.
static inline LogicalResult verifyFixedForm(Operation* op, Form fixedForm) {
  auto checkForm =
      [&](Form f) {
        if (f != fixedForm) {
          op->emitOpError()
              << "requires all polynomial operands and results to be in "
              << (fixedForm == Form::COEFF ? "coefficient" : "evaluation")
              << " form";
          return failure();
        }
        return success();
      };

  for (auto rTy : op->getResultTypes()) {
    if (auto pTy = dyn_cast<PolynomialType>(getElementTypeOrSelf(rTy))) {
      if (failed(checkForm(pTy.getForm()))) {
        return failure();
      }
    }
  }

  for (auto oTy : op->getOperandTypes()) {
    if (auto pTy = dyn_cast<PolynomialType>(getElementTypeOrSelf(oTy))) {
      if (failed(checkForm(pTy.getForm()))) {
        return failure();
      }
    }
  }
  return success();
}

// Trait that ensures that all operands and results polynomial have the same
// Form.
template <typename ConcreteType>
class SameOperandsAndResultForm
    : public OpTrait::TraitBase<ConcreteType, SameOperandsAndResultForm> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    std::optional<Form> form = std::nullopt;
    auto initOrCheckForm =
        [&](Form f) {
          if (!form.has_value()) {
            form = f;
            return success();
          }
          if (form.value() != f) {
            op->emitOpError()
                << "requires all polynomial operands and results to have "
                   "the same form";
            return failure();
          }
          return success();
        };

    for (auto rTy : op->getResultTypes()) {
      if (auto pTy = dyn_cast<PolynomialType>(getElementTypeOrSelf(rTy))) {
        if (failed(initOrCheckForm(pTy.getForm()))) {
          return failure();
        }
      }
    }

    for (auto oTy : op->getOperandTypes()) {
      if (auto pTy = dyn_cast<PolynomialType>(getElementTypeOrSelf(oTy))) {
        if (failed(initOrCheckForm(pTy.getForm()))) {
          return failure();
        }
      }
    }
    return success();
  }
};

// Trait that ensures that all operands and results polynomial are in COEFF
// form.
template <typename ConcreteType>
class FixedFormCoeff : public OpTrait::TraitBase<ConcreteType, FixedFormCoeff> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyFixedForm(op, Form::COEFF);
  }
};

// Trait that ensures that all operands and results polynomial are in EVAL form.
template <typename ConcreteType>
class FixedFormEval : public OpTrait::TraitBase<ConcreteType, FixedFormEval> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return verifyFixedForm(op, Form::EVAL);
  }
};

}  // namespace mlir::heir::polynomial

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTRAITS_H_
