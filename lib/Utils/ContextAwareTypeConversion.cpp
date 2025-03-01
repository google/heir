#include "lib/Utils/ContextAwareTypeConversion.h"

#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"       // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Convert types of the arguments and results of a function.
LogicalResult ContextAwareTypeConverter::convertFuncSignature(
    FunctionOpInterface funcOp, SmallVectorImpl<Type> &newArgTypes,
    SmallVectorImpl<Type> &newResultTypes) const {
  if (funcOp.isDeclaration()) {
    if (failed(convertTypes(funcOp.getArgumentTypes(), funcOp, newArgTypes)))
      return failure();
    if (failed(convertTypes(funcOp.getResultTypes(), funcOp, newResultTypes)))
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
  if (failed(typeConverter->convertTypes(op->getOperandTypes(),
                                         op->getOperands(), newOperandTypes)))
    return failure();

  SmallVector<Type> newResultTypes;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), op->getResults(),
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

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

void ContextAwareTypeConverter::SignatureConversion::addInputs(
    unsigned origInputNo, ArrayRef<Type> types) {
  assert(!types.empty() && "expected valid types");
  remapInput(origInputNo, /*newInputNo=*/argTypes.size(), types.size());
  addInputs(types);
}

void ContextAwareTypeConverter::SignatureConversion::addInputs(
    ArrayRef<Type> types) {
  assert(!types.empty() &&
         "1->0 type remappings don't need to be added explicitly");
  argTypes.append(types.begin(), types.end());
}

void ContextAwareTypeConverter::SignatureConversion::remapInput(
    unsigned origInputNo, unsigned newInputNo, unsigned newInputCount) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  assert(newInputCount != 0 && "expected valid input count");
  remappedInputs[origInputNo] =
      InputMapping{newInputNo, newInputCount, /*replacementValue=*/nullptr};
}

void ContextAwareTypeConverter::SignatureConversion::remapInput(
    unsigned origInputNo, Value replacementValue) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  remappedInputs[origInputNo] =
      InputMapping{origInputNo, /*size=*/0, replacementValue};
}

LogicalResult ContextAwareTypeConverter::convertType(
    Type t, Value v, SmallVectorImpl<Type> &results) const {
  assert(t && "expected non-null type");
  FailureOr<Attribute> result = getContextualAttr(v);
  if (failed(result)) return failure();
  Attribute attr = result.value();
  TypeAndAttribute key{t, attr};

  {
    std::shared_lock<decltype(cacheMutex)> cacheReadLock(cacheMutex,
                                                         std::defer_lock);
    if (t.getContext()->isMultithreadingEnabled()) cacheReadLock.lock();
    auto existingIt = cachedDirectConversions.find(key);
    if (existingIt != cachedDirectConversions.end()) {
      if (existingIt->second) results.push_back(existingIt->second);
      return success(existingIt->second != nullptr);
    }
  }
  // Walk the added converters in reverse order to apply the most recently
  // registered first.
  size_t currentCount = results.size();

  std::unique_lock<decltype(cacheMutex)> cacheWriteLock(cacheMutex,
                                                        std::defer_lock);

  for (const ConversionCallbackFn &converter : llvm::reverse(conversions)) {
    if (std::optional<LogicalResult> result = converter(t, v, results)) {
      if (t.getContext()->isMultithreadingEnabled()) cacheWriteLock.lock();
      if (!succeeded(*result)) {
        cachedDirectConversions.try_emplace(key, nullptr);
        return failure();
      }
      auto newTypes = ArrayRef<Type>(results).drop_front(currentCount);
      if (newTypes.size() == 1)
        cachedDirectConversions.try_emplace(key, newTypes.front());
      else
        cachedMultiConversions.try_emplace(key, llvm::to_vector<2>(newTypes));
      return success();
    }
  }
  return failure();
}

Type ContextAwareTypeConverter::convertType(Type t, Value v) const {
  // Use the multi-type result version to convert the type.
  SmallVector<Type, 1> results;
  if (failed(convertType(t, v, results))) return nullptr;

  // Check to ensure that only one type was produced.
  return results.size() == 1 ? results.front() : nullptr;
}

LogicalResult ContextAwareTypeConverter::convertTypes(
    TypeRange types, ValueRange values, SmallVectorImpl<Type> &results) const {
  for (const auto &[type, value] : llvm::zip(types, values))
    if (failed(convertType(type, value, results))) return failure();
  return success();
}

bool ContextAwareTypeConverter::isLegal(Type type, Value value) const {
  return convertType(type, value) == type;
}
bool ContextAwareTypeConverter::isLegal(TypeRange types,
                                        ValueRange values) const {
  return llvm::all_of(llvm::zip(types, values), [this](auto pair) {
    return isLegal(std::get<0>(pair), std::get<1>(pair));
  });
}
bool ContextAwareTypeConverter::isLegal(Operation *op) const {
  return isLegal(op->getOperandTypes(), op->getOperands()) &&
         isLegal(op->getResultTypes(), op->getResults());
}

bool ContextAwareTypeConverter::isLegal(Region *region) const {
  return llvm::all_of(*region, [this](Block &block) {
    return isLegal(block.getArgumentTypes(), block.getArguments());
  });
}

LogicalResult ContextAwareTypeConverter::convertSignatureArg(
    Operation *op, unsigned inputNo, Type type,
    SignatureConversion &result) const {
  // Try to convert the given input type.
  SmallVector<Type, 1> convertedTypes;
  if (failed(convertType(type, op, convertedTypes))) return failure();

  // If this argument is being dropped, there is nothing left to do.
  if (convertedTypes.empty()) return success();

  // Otherwise, add the new inputs.
  result.addInputs(inputNo, convertedTypes);
  return success();
}
LogicalResult ContextAwareTypeConverter::convertSignatureArgs(
    Operation *op, TypeRange types, SignatureConversion &result,
    unsigned origInputOffset) const {
  for (unsigned i = 0, e = types.size(); i != e; ++i)
    if (failed(convertSignatureArg(op, origInputOffset + i, types[i], result)))
      return failure();
  return success();
}

Value ContextAwareTypeConverter::materializeArgumentConversion(
    OpBuilder &builder, Location loc, Type resultType,
    ValueRange inputs) const {
  for (const MaterializationCallbackFn &fn :
       llvm::reverse(argumentMaterializations))
    if (Value result = fn(builder, resultType, inputs, loc)) return result;
  return nullptr;
}

Value ContextAwareTypeConverter::materializeSourceConversion(
    OpBuilder &builder, Location loc, Type resultType,
    ValueRange inputs) const {
  for (const MaterializationCallbackFn &fn :
       llvm::reverse(sourceMaterializations))
    if (Value result = fn(builder, resultType, inputs, loc)) return result;
  return nullptr;
}

Value ContextAwareTypeConverter::materializeTargetConversion(
    OpBuilder &builder, Location loc, Type resultType, ValueRange inputs,
    Type originalType) const {
  SmallVector<Value> result = materializeTargetConversion(
      builder, loc, TypeRange(resultType), inputs, originalType);
  if (result.empty()) return nullptr;
  assert(result.size() == 1 && "expected single result");
  return result.front();
}

SmallVector<Value> ContextAwareTypeConverter::materializeTargetConversion(
    OpBuilder &builder, Location loc, TypeRange resultTypes, ValueRange inputs,
    Type originalType) const {
  for (const TargetMaterializationCallbackFn &fn :
       llvm::reverse(targetMaterializations)) {
    SmallVector<Value> result =
        fn(builder, resultTypes, inputs, loc, originalType);
    if (result.empty()) continue;
    assert(TypeRange(ValueRange(result)) == resultTypes &&
           "callback produced incorrect number of values or values with "
           "incorrect types");
    return result;
  }
  return {};
}

std::optional<ContextAwareTypeConverter::SignatureConversion>
ContextAwareTypeConverter::convertBlockSignature(Block *block) const {
  SignatureConversion conversion(block->getNumArguments());
  if (failed(convertSignatureArgs(block->getParentOp(),
                                  block->getArgumentTypes(), conversion)))
    return std::nullopt;
  return conversion;
}

//===----------------------------------------------------------------------===//
// Type attribute conversion
//===----------------------------------------------------------------------===//
ContextAwareTypeConverter::AttributeConversionResult
ContextAwareTypeConverter::AttributeConversionResult::result(Attribute attr) {
  return AttributeConversionResult(attr, resultTag);
}

ContextAwareTypeConverter::AttributeConversionResult
ContextAwareTypeConverter::AttributeConversionResult::na() {
  return AttributeConversionResult(nullptr, naTag);
}

ContextAwareTypeConverter::AttributeConversionResult
ContextAwareTypeConverter::AttributeConversionResult::abort() {
  return AttributeConversionResult(nullptr, abortTag);
}

bool ContextAwareTypeConverter::AttributeConversionResult::hasResult() const {
  return impl.getInt() == resultTag;
}

bool ContextAwareTypeConverter::AttributeConversionResult::isNa() const {
  return impl.getInt() == naTag;
}

bool ContextAwareTypeConverter::AttributeConversionResult::isAbort() const {
  return impl.getInt() == abortTag;
}

Attribute ContextAwareTypeConverter::AttributeConversionResult::getResult()
    const {
  assert(hasResult() && "Cannot get result from N/A or abort");
  return impl.getPointer();
}

std::optional<Attribute> ContextAwareTypeConverter::convertTypeAttribute(
    Type type, Attribute attr) const {
  for (const TypeAttributeConversionCallbackFn &fn :
       llvm::reverse(typeAttributeConversions)) {
    AttributeConversionResult res = fn(type, attr);
    if (res.hasResult()) return res.getResult();
    if (res.isAbort()) return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace heir
}  // namespace mlir
