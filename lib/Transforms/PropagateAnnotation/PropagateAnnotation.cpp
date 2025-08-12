#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Iterators.h"     // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"      // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "propagate-annotation"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PROPAGATEANNOTATION
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h.inc"

static void setAttrIfMissing(Value value, StringRef attrName, Attribute attr) {
  if (failed(findAttributeAssociatedWith(value, attrName))) {
    setAttributeAssociatedWith(value, attrName, attr);
  }
}

void forwardPropagateAnnotation(Operation* root, StringRef attrName,
                                function_ref<bool(Type)> shouldPropagate) {
  LLVM_DEBUG(llvm::dbgs() << "Forward propagation of " << attrName << "\n");
  if (attrName.empty()) {
    return;
  }
  root->walk<WalkOrder::PreOrder>([&](Operation* op) {
    LLVM_DEBUG(llvm::dbgs() << "Visiting op " << op->getName() << "\n");
    if (op->hasAttr(attrName)) {
      return WalkResult::advance();
    }

    // A terminator propagates to parent's result attrs, if the op implements
    // the OperandAndResultAttrInterface or is a func.func.
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      LLVM_DEBUG(llvm::dbgs() << "Op is a terminator\n");
      // Terminators try to inherit their attribute from the parent op's return
      // attr.
      auto* parentOp = op->getParentOp();

      // An op like `return %0, %1` propagates multiple attributes forward to
      // the parent's multiple result attrs.
      for (int i = 0; i < op->getNumOperands(); ++i) {
        auto operand = op->getOperand(i);
        FailureOr<Attribute> attr =
            findAttributeAssociatedWith(operand, attrName);
        if (failed(attr)) continue;
        if (!shouldPropagate(operand.getType())) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping propagation of " << attrName
                                  << " to operand " << operand << "\n");
          continue;
        }

        llvm::TypeSwitch<Operation*>(parentOp)
            .Case<FunctionOpInterface, OperandAndResultAttrInterface>(
                [&](auto interface) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "Propagating terminator operand attr " << i
                             << " (" << attr << ") to parent result attr " << i
                             << "\n");
                  return interface.setResultAttr(i, attrName, attr.value());
                })
            .Default([&](Operation* op) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Warning: propagating terminator operand attr " << i
                         << " (" << attr
                         << ") to unsupported parent result attr\n");
              return op->setAttr(attrName, attr.value());
            });
      }
      return WalkResult::advance();
    }

    // Otherwise, operands propagate to result attributes of the op, unless
    // the results should not be propagated to.
    if (llvm::all_of(op->getResultTypes(),
                     [&](Type type) { return !shouldPropagate(type); })) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping propagation of " << attrName
                              << " through op " << op->getName() << "\n");
      return WalkResult::advance();
    }

    for (Value operand : op->getOperands()) {
      Attribute attr;
      if (dyn_cast<BlockArgument>(operand)) {
        auto res = findAttributeAssociatedWith(operand, attrName);
        if (failed(res)) {
          continue;
        }
        attr = *res;
      } else {
        // findAttributeAssociatedWith works not so well for ArrayAttr
        attr = operand.getDefiningOp()->getAttr(attrName);
        if (!attr) {
          continue;
        }
      }

      op->setAttr(attrName, attr);
      // short-circuit if we found an attribute
      break;
    }
    return WalkResult::advance();
  });
}

void backwardPropagateAnnotation(Operation* root, StringRef attrName,
                                 function_ref<bool(Type)> shouldPropagate) {
  if (attrName.empty()) {
    return;
  }

  root->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation* op) {
    LLVM_DEBUG(llvm::dbgs() << "BackProp(" << attrName << ") visiting op "
                            << op->getName() << "\n");

    if (op->hasAttr(attrName)) {
      // The attr is assumed to be associated with the op's results.
      Attribute attrToPropagate = op->getAttr(attrName);
      LLVM_DEBUG(llvm::dbgs()
                 << "Using op's result attr " << attrToPropagate << "\n");
      for (auto operand : op->getOperands()) {
        if (!shouldPropagate(operand.getType())) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping propagation of " << attrName
                                  << " to operand " << operand << "\n");
          continue;
        }
        setAttrIfMissing(operand, attrName, attrToPropagate);
      }
      return WalkResult::advance();
    }

    if (auto attrInterface = dyn_cast<OperandAndResultAttrInterface>(op)) {
      // The attr is attached to the op's operands and can be propagated
      // backward to the op's defining op.
      for (OpOperand& operand : op->getOpOperands()) {
        if (!shouldPropagate(operand.get().getType())) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping propagation of " << attrName
                                  << " to operand " << operand.get() << "\n");
          continue;
        }
        auto attr =
            attrInterface.getOperandAttr(operand.getOperandNumber(), attrName);
        if (attr) {
          setAttrIfMissing(operand.get(), attrName, attr);
        }
      }
    }

    if (op->hasTrait<OpTrait::IsTerminator>()) {
      LLVM_DEBUG(llvm::dbgs() << "Op is a terminator\n");
      // Terminators try to inherit their attribute from the parent op's return
      // attr.
      auto* parentOp = op->getParentOp();

      // An op like `return %0, %1` can inherit multiple attributes from the
      // parent's multiple result attrs.
      for (int i = 0; i < op->getNumOperands(); ++i) {
        Attribute attr =
            llvm::TypeSwitch<Operation*, Attribute>(parentOp)
                .Case<FunctionOpInterface, OperandAndResultAttrInterface>(
                    [&](auto interface) {
                      return interface.getResultAttr(i, attrName);
                    })
                .Default([&](Operation* op) { return op->getAttr(attrName); });
        if (attr) {
          auto operand = op->getOperand(i);
          if (!shouldPropagate(operand.getType())) {
            LLVM_DEBUG(llvm::dbgs() << "Skipping propagation of " << attrName
                                    << " to operand " << operand << "\n");
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "Propagating result attr " << i << " ("
                                  << attr << ") to operand " << i << "\n");
          setAttrIfMissing(operand, attrName, attr);
        }
      }
      return WalkResult::advance();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op because no suitable cases found.\n");
    return WalkResult::advance();
  });
}

struct PropagateAnnotation
    : impl::PropagateAnnotationBase<PropagateAnnotation> {
  using PropagateAnnotationBase::PropagateAnnotationBase;

  void runOnOperation() override {
    if (reverse) {
      backwardPropagateAnnotation(getOperation(), attrName);
    } else {
      forwardPropagateAnnotation(getOperation(), attrName);
    }
  }
};

}  // namespace heir
}  // namespace mlir
