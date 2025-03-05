#include "lib/Utils/AttributeUtils.h"

#include <optional>
#include <string>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project

namespace mlir {
namespace heir {

Attribute getAttributeFromValue(Value value, StringRef attrName) {
  Attribute attr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto genericOp = dyn_cast<secret::GenericOp>(parentOp);
    if (genericOp) {
      attr = genericOp.getArgAttr(blockArg.getArgNumber(), attrName);
    }
    auto funcOp = dyn_cast<func::FuncOp>(parentOp);
    if (funcOp) {
      attr = funcOp.getArgAttr(blockArg.getArgNumber(), attrName);
    }
  } else {
    auto *parentOp = value.getDefiningOp();
    attr = parentOp->getAttr(attrName);
  }
  return attr;
}

void setAttributeForValue(Value value, StringRef attrName, Attribute attr) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto genericOp = dyn_cast<secret::GenericOp>(parentOp);
    if (genericOp) {
      genericOp.setArgAttr(blockArg.getArgNumber(), attrName, attr);
    }
    auto funcOp = dyn_cast<func::FuncOp>(parentOp);
    if (funcOp) {
      funcOp.setArgAttr(blockArg.getArgNumber(), attrName, attr);
    }
  } else {
    auto *parentOp = value.getDefiningOp();
    parentOp->setAttr(attrName, attr);
  }
}

}  // namespace heir
}  // namespace mlir
