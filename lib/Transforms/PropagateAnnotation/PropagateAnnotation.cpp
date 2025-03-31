#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"

#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PROPAGATEANNOTATION
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h.inc"

struct PropagateAnnotation
    : impl::PropagateAnnotationBase<PropagateAnnotation> {
  using PropagateAnnotationBase::PropagateAnnotationBase;

  void runOnOperation() override {
    if (attrName.empty()) {
      return;
    }
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
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
};

}  // namespace heir
}  // namespace mlir
