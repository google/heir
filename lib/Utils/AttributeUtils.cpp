#include "lib/Utils/AttributeUtils.h"

#include <cassert>

#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

int getOperandNumber(Operation *op, Value value) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get() == value) return operand.getOperandNumber();
  }
  return -1;
}

Attribute findAttributeForBlockArgument(BlockArgument blockArg,
                                        StringRef attrName) {
  auto *parentOp = blockArg.getOwner()->getParentOp();
  assert(parentOp != nullptr &&
         "Missing parent op! Was this value not properly remapped?");

  return llvm::TypeSwitch<Operation *, Attribute>(parentOp)
      .Case<FunctionOpInterface>([&](auto op) -> Attribute {
        return op.getArgAttr(blockArg.getArgNumber(), attrName);
      })
      // AffineForOp needs a special case because its operands do not
      // line up perfectly with its block arguments.
      // This could be potentially improved by adding an interface method to
      // OperandAndResultAttrInterface to map from block arguments to operands
      .Case<affine::AffineForOp>([&](affine::AffineForOp op) -> Attribute {
        // For op has as its first block argument the induction
        // variable, which does not correspond to a single operand.
        if (blockArg.getArgNumber() == 0) return nullptr;

        auto argAttrInterface = cast<OperandAndResultAttrInterface>(*op);
        auto initArg = op.getInits()[blockArg.getArgNumber() - 1];
        int operandNumber = getOperandNumber(op, initArg);
        if (operandNumber == -1) return nullptr;
        return argAttrInterface.getOperandAttr(operandNumber, attrName);
      })
      .Case<OperandAndResultAttrInterface>([&](auto op) -> Attribute {
        return op.getOperandAttr(blockArg.getArgNumber(), attrName);
      })
      .Default([](Operation *op) { return nullptr; });
}

Attribute findAttributeForResult(Value result, StringRef attrName) {
  auto *parentOp = result.getDefiningOp();
  assert(parentOp &&
         "Missing defining op, but the input value is not a BlockArgument. "
         "This is madness.");

  return llvm::TypeSwitch<Operation *, Attribute>(parentOp)
      .Case<OperandAndResultAttrInterface>([&](auto op) -> Attribute {
        int resultNumber = cast<OpResult>(result).getResultNumber();
        if (resultNumber == -1) return nullptr;
        Attribute attr = op.getResultAttr(resultNumber, attrName);
        if (attr) return attr;
        return op->getAttr(attrName);
      })
      .Default([&](Operation *op) { return op->getAttr(attrName); });
}

FailureOr<Attribute> findAttributeAssociatedWith(Value value,
                                                 StringRef attrName) {
  Attribute attr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    attr = findAttributeForBlockArgument(blockArg, attrName);
  } else {
    attr = findAttributeForResult(value, attrName);
  }

  if (!attr) return failure();

  return attr;
}

void clearAttrs(Operation *op, StringRef attrName) {
  op->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<FunctionOpInterface>([&](auto op) {
          for (auto i = 0; i != op.getNumArguments(); ++i) {
            op.removeArgAttr(i, attrName);
          }
          for (auto i = 0; i != op.getNumResults(); ++i) {
            op.removeResultAttr(i, StringAttr::get(op->getContext(), attrName));
          }
        })
        .Case<OperandAndResultAttrInterface>([&](auto op) {
          for (auto i = 0; i != op->getNumOperands(); ++i) {
            op.removeOperandAttr(i, attrName);
          }
          for (auto i = 0; i != op->getNumResults(); ++i) {
            op.removeResultAttr(i, attrName);
          }
          op->removeAttr(attrName);
        })
        .Default([&](Operation *op) { op->removeAttr(attrName); });
  });
}

void copyReturnOperandAttrsToFuncResultAttrs(Operation *op,
                                             StringRef attrName) {
  op->walk([&](func::FuncOp funcOp) {
    for (auto &block : funcOp.getBlocks()) {
      for (OpOperand &returnOperand : block.getTerminator()->getOpOperands()) {
        FailureOr<Attribute> attr =
            findAttributeAssociatedWith(returnOperand.get(), attrName);
        if (failed(attr)) continue;
        funcOp.setResultAttr(returnOperand.getOperandNumber(), attrName,
                             attr.value());
      }
    }
  });
}

void populateOperandAttrInterface(Operation *op, StringRef attrName) {
  op->walk<WalkOrder::PreOrder>([&](OperandAndResultAttrInterface opInt) {
    for (auto &opOperand : opInt->getOpOperands()) {
      FailureOr<Attribute> attrResult =
          findAttributeAssociatedWith(opOperand.get(), attrName);

      if (failed(attrResult)) continue;

      opInt.setOperandAttr(opOperand.getOperandNumber(), attrName,
                           attrResult.value());
    }
  });
}

}  // namespace heir
}  // namespace mlir
