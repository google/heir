#include "lib/Utils/ContextAwareTypeConversion.h"

#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"       // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool ContextAwareTypeConverter::isLegal(Operation *op) const {
  SmallVector<Type> newOperandTypes;
  if (failed(convertValueRangeTypes(op->getOperands(), newOperandTypes)))
    return false;

  SmallVector<Type> newResultTypes;
  if (failed(convertValueRangeTypes(op->getResults(), newResultTypes)))
    return false;

  return op->getOperandTypes() == newOperandTypes &&
         op->getResultTypes() == newResultTypes;
}

bool ContextAwareTypeConverter::isLegal(FunctionOpInterface funcOp) const {
  SmallVector<Type> newOperandTypes;
  SmallVector<Type> newResultTypes;

  if (failed(convertFuncSignature(funcOp, newOperandTypes, newResultTypes)))
    return false;

  return funcOp.getArgumentTypes() == ArrayRef<Type>(newOperandTypes) &&
         funcOp.getResultTypes() == ArrayRef<Type>(newResultTypes);
}

// Convert a range of values, with converted types stored in newTypes.
LogicalResult AttributeAwareTypeConverter::convertValueRangeTypes(
    ValueRange values, SmallVectorImpl<Type> &newTypes) const {
  newTypes.reserve(values.size());
  for (auto value : values) {
    FailureOr<Attribute> attr = getContextualAttr(value);
    // If no contextual attribute is found, it may be a type that doesn't need
    // conversion. In this case, just use the type as is. An example of this is
    // a bgv.rotate op, which consumes a ciphertext (which can be converted)
    // and an index to rotate (which needs no conversion).
    if (failed(attr)) {
      newTypes.push_back(value.getType());
      continue;
    }

    FailureOr<Type> newType = convert(value.getType(), attr.value());
    if (failed(newType)) return failure();

    newTypes.push_back(newType.value());
  }

  return success();
}

// Convert types of the arguments and results of a function.
LogicalResult AttributeAwareTypeConverter::convertFuncSignature(
    FunctionOpInterface funcOp, SmallVectorImpl<Type> &newArgTypes,
    SmallVectorImpl<Type> &newResultTypes) const {
  if (funcOp.isDeclaration()) {
    if (failed(convertTypes(funcOp.getArgumentTypes(), newArgTypes)))
      return failure();
    if (failed(convertTypes(funcOp.getResultTypes(), newResultTypes)))
      return failure();
    return success();
  }

  for (auto argument : funcOp.getArguments()) {
    FailureOr<Attribute> attr = getContextualAttr(argument);
    if (failed(attr)) {
      newArgTypes.push_back(argument.getType());
      continue;
    }
    FailureOr<Type> newType = convert(argument.getType(), attr.value());
    if (failed(newType)) return failure();
    newArgTypes.push_back(newType.value());
  }
  // To get the value corresponding to the func's return types, we need to get
  // the terminator operands.
  for (auto &block : funcOp.getBlocks()) {
    for (auto result : block.getTerminator()->getOperands()) {
      FailureOr<Attribute> attr = getContextualAttr(result);
      if (failed(attr)) return failure();
      FailureOr<Type> newType = convert(result.getType(), attr.value());
      if (failed(newType)) return failure();
      newResultTypes.push_back(newType.value());
    }
  }
  return success();
}

LogicalResult ConvertFuncWithContextAwareTypeConverter::matchAndRewrite(
    func::FuncOp funcOp, PatternRewriter &rewriter) const {
  SmallVector<Type> oldFuncOperandTypes(funcOp.getFunctionType().getInputs());
  SmallVector<Type> oldFuncResultTypes(funcOp.getFunctionType().getResults());
  SmallVector<Type> newFuncOperandTypes;
  SmallVector<Type> newFuncResultTypes;
  if (failed(contextAwareTypeConverter->convertFuncSignature(
          funcOp, newFuncOperandTypes, newFuncResultTypes)))
    return funcOp->emitError("Failed to convert function signature");

  auto newFuncType =
      FunctionType::get(getContext(), newFuncOperandTypes, newFuncResultTypes);
  rewriter.modifyOpInPlace(funcOp, [&] {
    funcOp.setType(newFuncType);

    if (funcOp.isDeclaration()) return;

    // Set the block argument types to match the new signature
    for (auto [arg, newType] : llvm::zip(
             funcOp.getBody().front().getArguments(), newFuncOperandTypes)) {
      arg.setType(newType);
    }

    // This is a weird part related to the hacky nature of this context-aware
    // type conversion. In order to make this work, we have to also modify the
    // func.return at the same time as the func.func containing it. Otherwise,
    // if we tried to make a separate "context aware" conversion pattern for
    // the func.return op, it would not have the type-converted operands
    // available to its OpAdaptor. Furthermore, when I tried making a separate
    // context-aware pattern for func.return in isolation, I couldn't get it to
    // legalize and the conversion engine looped infinitely.
    Block &block = funcOp.getBody().front();
    for (auto [returnOperand, newType] :
         llvm::zip(block.getTerminator()->getOperands(), newFuncResultTypes)) {
      returnOperand.setType(newType);
    }
  });

  if (failed(finalizeFuncOpModification(funcOp, oldFuncOperandTypes,
                                        oldFuncResultTypes, rewriter)))
    return failure();

  return success();
}

LogicalResult convertAnyOperand(const ContextAwareTypeConverter *typeConverter,
                                Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) {
  if (typeConverter->isLegal(op)) {
    return failure();
  }

  SmallVector<Type> newOperandTypes;
  if (failed(typeConverter->convertValueRangeTypes(op->getOperands(),
                                                   newOperandTypes)))
    return failure();

  SmallVector<Type> newResultTypes;
  if (failed(typeConverter->convertValueRangeTypes(op->getResults(),
                                                   newResultTypes)))
    return failure();

  SmallVector<std::unique_ptr<Region>, 1> regions;
  if (!op->getRegions().empty()) {
    // Because the Dialect conversion framework handles converting region types
    // and it requires some extra work supporting block signature type
    // conversion, etc. Do this when the need arises.
    return op->emitError(
        "Generic context-aware op conversion requires op to have no regions");
  }

  Operation *newOp = rewriter.create(OperationState(
      op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
      op->getAttrs(), op->getSuccessors(), regions));

  rewriter.replaceOp(op, newOp);
  return success();
}

}  // namespace heir
}  // namespace mlir
