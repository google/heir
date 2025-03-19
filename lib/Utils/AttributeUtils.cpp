#include "lib/Utils/AttributeUtils.h"

#include <cassert>

#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
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

Attribute getUndistinguishedResultAttr(Operation *op, int resultNumber,
                                       StringRef attrName) {
  Attribute attr = op->getAttr(attrName);
  // Some ops store the relevant attribute as an array attr with a name
  // (e.g., "tensor_ext.layout" being an array of layout attrs for a
  // multi-result op. In this case, we assert the array entries match
  // with operation results, and return the relevant entry.
  if (auto arrayAttr = dyn_cast_or_null<ArrayAttr>(attr)) {
    assert(arrayAttr.size() == op->getNumResults() &&
           "Array attr does not match number of results");
    return arrayAttr[resultNumber];
  }
  return attr;
}

Attribute findAttributeForResult(Value result, StringRef attrName) {
  auto *parentOp = result.getDefiningOp();
  assert(parentOp &&
         "Missing defining op, but the input value is not a BlockArgument. "
         "This is madness.");
  int resultNumber = cast<OpResult>(result).getResultNumber();

  return llvm::TypeSwitch<Operation *, Attribute>(parentOp)
      .Case<OperandAndResultAttrInterface>([&](auto op) -> Attribute {
        Attribute attr = op.getResultAttr(resultNumber, attrName);
        if (attr) return attr;
        return getUndistinguishedResultAttr(op, resultNumber, attrName);
      })
      .Default([&](Operation *op) {
        return getUndistinguishedResultAttr(op, resultNumber, attrName);
      });
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

void setAttributeForBlockArgument(BlockArgument blockArg, StringRef attrName,
                                  Attribute attr) {
  auto *parentOp = blockArg.getOwner()->getParentOp();
  assert(parentOp != nullptr &&
         "Missing parent op! Was this value not properly remapped?");

  llvm::TypeSwitch<Operation *>(parentOp)
      .Case<FunctionOpInterface>([&](auto op) {
        return op.setArgAttr(blockArg.getArgNumber(), attrName, attr);
      })
      .Case<affine::AffineForOp>([&](affine::AffineForOp op) {
        // For op has as its first block argument the induction
        // variable, which does not correspond to a single operand.
        if (blockArg.getArgNumber() == 0) return;

        auto argAttrInterface = cast<OperandAndResultAttrInterface>(*op);
        auto initArg = op.getInits()[blockArg.getArgNumber() - 1];
        int operandNumber = getOperandNumber(op, initArg);
        if (operandNumber == -1) return;
        return argAttrInterface.setOperandAttr(operandNumber, attrName, attr);
      })
      .Case<OperandAndResultAttrInterface>([&](auto op) {
        return op.setOperandAttr(blockArg.getArgNumber(), attrName, attr);
      });
}

void setAttributeForResult(Value result, StringRef attrName, Attribute attr) {
  auto *parentOp = result.getDefiningOp();
  assert(parentOp &&
         "Missing defining op, but the input value is not a BlockArgument. "
         "This is madness.");
  int resultNumber = cast<OpResult>(result).getResultNumber();

  llvm::TypeSwitch<Operation *>(parentOp)
      .Case<OperandAndResultAttrInterface>(
          [&](auto op) { op.setResultAttr(resultNumber, attrName, attr); })
      .Default([&](Operation *op) {
        // If the op has an existing array attr, set the attribute in the
        // existing array attr's index.
        auto existingAttr = op->getAttr(attrName);
        if (auto existingArrayAttr =
                dyn_cast_or_null<ArrayAttr>(existingAttr)) {
          assert(existingArrayAttr.size() == op->getNumResults() &&
                 "Array attr does not match number of results");
          existingArrayAttr[resultNumber] = attr;
          op->setAttr(attrName, existingArrayAttr);
        } else {
          assert(op->getNumResults() == 1 &&
                 "Op has multiple results but no array attr was passed or "
                 "existing.");
          op->setAttr(attrName, attr);
        }
      });
}

void setAttributeAssociatedWith(Value value, StringRef attrName,
                                Attribute attr) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    setAttributeForBlockArgument(blockArg, attrName, attr);
  } else {
    setAttributeForResult(value, attrName, attr);
  }
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
          op->removeAttr(attrName);
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
