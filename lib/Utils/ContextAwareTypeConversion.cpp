#include "lib/Utils/ContextAwareTypeConversion.h"

#include <cassert>
#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>

#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define DEBUG_TYPE "context-aware-type-conversion"

namespace mlir {
namespace heir {

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
  auto contextAttr = getContextualAttr(v);
  // If no usable attribute is found, then we don't need to convert the type,
  // and we can return the type as-is
  if (failed(contextAttr)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to fetch attr\n");
    results.push_back(t);
    return success();
  }
  LLVM_DEBUG(llvm::dbgs() << "Fetched attr: " << contextAttr.value() << "\n");
  return convertType(t, contextAttr.value(), results);
}

LogicalResult ContextAwareTypeConverter::convertType(
    Type t, Attribute attr, SmallVectorImpl<Type> &results) const {
  assert(t && "expected non-null type");
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
    if (std::optional<LogicalResult> result = converter(t, attr, results)) {
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

Type ContextAwareTypeConverter::convertType(Type t, Attribute attr) const {
  // Use the multi-type result version to convert the type.
  SmallVector<Type, 1> results;
  if (failed(convertType(t, attr, results))) return nullptr;

  // Check to ensure that only one type was produced.
  return results.size() == 1 ? results.front() : nullptr;
}

LogicalResult ContextAwareTypeConverter::convertTypes(
    TypeRange types, ArrayRef<Attribute> attributes,
    SmallVectorImpl<Type> &results) const {
  for (const auto &[type, attr] : llvm::zip(types, attributes))
    if (failed(convertType(type, attr, results))) return failure();
  return success();
}

LogicalResult ContextAwareTypeConverter::convertTypes(
    TypeRange types, ValueRange values, SmallVectorImpl<Type> &results) const {
  for (const auto &[type, value] : llvm::zip(types, values)) {
    auto resultAttr = getContextualAttr(value);
    if (failed(resultAttr)) {
      // If the attribute cannot be found, the does not need to be converted
      results.push_back(type);
      continue;
    }
    if (failed(convertType(type, resultAttr.value(), results)))
      return failure();
  }
  return success();
}

bool ContextAwareTypeConverter::isLegal(Type type, Attribute attr) const {
  return convertType(type, attr) == type;
}
bool ContextAwareTypeConverter::isLegal(TypeRange types,
                                        ArrayRef<Attribute> attributes) const {
  return llvm::all_of(llvm::zip(types, attributes), [this](auto pair) {
    return isLegal(std::get<0>(pair), std::get<1>(pair));
  });
}
bool ContextAwareTypeConverter::isLegal(Operation *op) const {
  // HEIR: this is a hack to ensure that the attribute lookup will not fail.
  // If the attribute lookup fails, it may be because the block containing
  // this op has been unlinked and the values not yet remapped. A
  // ConvertAny<> should resolve an op that is illegal because of this.
  bool maySegfaultOnAttrLookup =
      llvm::any_of(op->getOperands(), [](Value value) {
        if (isa<BlockArgument>(value)) {
          return cast<BlockArgument>(value).getOwner()->getParentOp() ==
                 nullptr;
        }
        return value.getDefiningOp() == nullptr;
      });

  if (maySegfaultOnAttrLookup) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping isLegal check for op with potentially unlinked "
                  "values\n");
    return false;
  }

  SmallVector<Attribute> operandAttrs;
  SmallVector<Type> operandTypes;
  for (auto operand : op->getOperands()) {
    auto result = getContextualAttr(operand);
    if (succeeded(result)) {
      operandAttrs.push_back(result.value());
      operandTypes.push_back(operand.getType());
    }
  }
  SmallVector<Attribute> resultAttrs;
  SmallVector<Type> resultTypes;
  for (auto value : op->getResults()) {
    auto res = getContextualAttr(value);
    if (succeeded(res)) {
      resultAttrs.push_back(res.value());
      resultTypes.push_back(value.getType());
    }
  }
  return isLegal(operandTypes, operandAttrs) &&
         isLegal(resultTypes, resultAttrs);
}

bool ContextAwareTypeConverter::isLegal(Region *region) const {
  return llvm::all_of(*region, [this](Block &block) {
    auto argumentAttrs = llvm::map_to_vector(
        block.getArguments(),
        [this](Value v) { return getContextualAttr(v).value_or(nullptr); });
    return isLegal(block.getArgumentTypes(), argumentAttrs);
  });
}

bool ContextAwareTypeConverter::isSignatureLegal(
    FunctionOpInterface funcOp) const {
  SmallVector<Type> newArgTypes;
  SmallVector<Type> newResultTypes;
  if (failed(convertFuncSignature(funcOp, newArgTypes, newResultTypes)))
    return false;

  return newArgTypes == funcOp.getArgumentTypes() &&
         newResultTypes == funcOp.getResultTypes();
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
  // HEIR: don't reuse convertSignatureArgs, instead get the context
  // value and convert the types directly. This is because the code
  // cannot be shared with FuncOp type conversion anymore (blocks have
  // SSA values for context, but FuncOps may not).
  SignatureConversion conversion(block->getNumArguments());
  auto values = block->getArguments();
  auto types = block->getArgumentTypes();
  for (unsigned i = 0, e = types.size(); i != e; ++i) {
    SmallVector<Type, 1> convertedTypes;
    if (failed(convertType(types[i], values[i], convertedTypes)))
      return std::nullopt;

    // If this argument is being dropped, there is nothing left to do.
    if (convertedTypes.empty()) continue;

    // Otherwise, add the new inputs.
    conversion.addInputs(i, convertedTypes);
  }
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
