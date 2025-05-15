#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Iterators.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "propagate-annotation"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PROPAGATEANNOTATION
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h.inc"

void forwardPropagateAnnotation(Operation *root, StringRef attrName) {
  if (attrName.empty()) {
    return;
  }
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->hasAttr(attrName)) {
      return;
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
      return;
    }
  });
}

void setAttrIfMissing(Value value, StringRef attrName, Attribute attr) {
  if (failed(findAttributeAssociatedWith(value, attrName))) {
    setAttributeAssociatedWith(value, attrName, attr);
  }
}

void backwardPropagateAnnotation(Operation *root, StringRef attrName) {
  if (attrName.empty()) {
    return;
  }

  root->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs() << "BackProp(" << attrName << ") visiting op "
                            << op->getName() << "\n");
    if (op->hasAttr(attrName)) {
      // The attr is assumed to be associated with the op's results.
      Attribute attrToPropagate = op->getAttr(attrName);
      LLVM_DEBUG(llvm::dbgs()
                 << "Using op's result attr " << attrToPropagate << "\n");
      for (auto operand : op->getOperands()) {
        setAttrIfMissing(operand, attrName, attrToPropagate);
      }
      return WalkResult::advance();
    }

    if (op->hasTrait<OpTrait::IsTerminator>()) {
      LLVM_DEBUG(llvm::dbgs() << "Op is a terminator\n");
      // Terminators try to inherit their attribute from the parent op's return
      // attr.
      auto *parentOp = op->getParentOp();

      // An op like `return %0, %1` can inherit multiple attributes from the
      // parent's multiple result attrs.
      for (int i = 0; i < op->getNumOperands(); ++i) {
        Attribute attr =
            llvm::TypeSwitch<Operation *, Attribute>(parentOp)
                .Case<FunctionOpInterface, OperandAndResultAttrInterface>(
                    [&](auto interface) {
                      return interface.getResultAttr(i, attrName);
                    })
                .Default([&](Operation *op) { return op->getAttr(attrName); });
        if (attr) {
          LLVM_DEBUG(llvm::dbgs() << "Propagating result attr " << i << " ("
                                  << attr << ") to operand " << i << "\n");
          setAttrIfMissing(op->getOperand(i), attrName, attr);
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
