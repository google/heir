#include "lib/Utils/ContextAwareConversionUtils.h"

#include <iterator>
#include <string>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Convert the types in a function signature using the relevant attributes and a
// custom converter once the attribute is found. Returns true if the signature
// changed.
bool convertFuncSignatureUsingAttrs(FunctionOpInterface funcOp,
                                    SmallVectorImpl<Type>& newArgTypes,
                                    SmallVectorImpl<Type>& newResultTypes,
                                    StringRef attrName,
                                    CustomTypeConverter converterFn) {
  bool changed = false;
  for (int i = 0; i < funcOp.getNumArguments(); ++i) {
    auto argType = funcOp.getArgumentTypes()[i];
    auto contextAttr = funcOp.getArgAttr(i, attrName);
    if (!contextAttr) {
      newArgTypes.push_back(argType);
      continue;
    }

    auto convertedType = converterFn(argType, contextAttr);
    if (!convertedType.has_value()) {
      newResultTypes.push_back(argType);
      continue;
    }
    newArgTypes.push_back(convertedType.value());
    changed = true;
  }

  for (int i = 0; i < funcOp.getNumResults(); ++i) {
    auto resultType = funcOp.getResultTypes()[i];
    auto contextAttr = funcOp.getResultAttr(i, attrName);
    if (!contextAttr) {
      newResultTypes.push_back(resultType);
      continue;
    }

    auto convertedType = converterFn(resultType, contextAttr);
    if (!convertedType.has_value()) {
      newResultTypes.push_back(resultType);
      continue;
    }
    newResultTypes.push_back(convertedType.value());
    changed = true;
  }

  return changed;
}

LogicalResult ContextAwareFuncConversion::matchAndRewrite(
    func::FuncOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  FunctionType type = dyn_cast<FunctionType>(op.getFunctionType());
  if (!type) return failure();

  // Convert the original function types.
  TypeConverter::SignatureConversion result(type.getNumInputs());

  SmallVector<Type, 4> convertedArgTypes, convertedResultTypes;
  bool changed = convertFuncSignatureUsingAttrs(
      cast<FunctionOpInterface>(op.getOperation()), convertedArgTypes,
      convertedResultTypes, attrName, customTypeConverter);

  // This is used by the later convertRegionTypes
  result.addInputs(convertedArgTypes);

  if (!changed) return failure();

  if (!op.isDeclaration() &&
      failed(rewriter.convertRegionTypes(&op.getFunctionBody(),
                                         *getTypeConverter(), &result)))
    return failure();

  // Update the function signature in-place.
  auto newType = FunctionType::get(rewriter.getContext(), convertedArgTypes,
                                   convertedResultTypes);

  SmallVector<Type> oldFuncOperandTypes(type.getInputs());
  SmallVector<Type> oldFuncResultTypes(type.getResults());

  rewriter.modifyOpInPlace(op, [&] { op.setType(newType); });

  if (failed(finalizeFuncOpModification(op, oldFuncOperandTypes,
                                        oldFuncResultTypes, rewriter)))
    return failure();
  return success();
}

FailureOr<Operation*> SecretGenericFuncCallConversion::matchAndRewriteInner(
    secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
    ArrayRef<NamedAttribute> attributes,
    ConversionPatternRewriter& rewriter) const {
  // check if any args are secret from wrapping generic
  // clone the callee (and update a unique name, for now always) the call
  // operands add a note that we don't have to always clone to be secret
  // update the called func's type signature

  func::CallOp callOp = *op.getBody()->getOps<func::CallOp>().begin();
  auto module = callOp->getParentOfType<ModuleOp>();
  func::FuncOp callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());

  // For now, ensure that there is only one caller to the callee.
  auto calleeUses = callee.getSymbolUses(callee->getParentOp());
  if (std::distance(calleeUses->begin(), calleeUses->end()) != 1) {
    return op->emitError() << "expected exactly one caller to the callee";
  }

  SmallVector<Type> newInputTypes;
  for (auto val : inputs) {
    newInputTypes.push_back(val.getType());
  }

  FunctionType newFunctionType =
      cast<FunctionType>(callee.cloneTypeWith(newInputTypes, outputTypes));
  auto newFuncOp = rewriter.cloneWithoutRegions(callee);
  newFuncOp->moveAfter(callee);
  newFuncOp.setFunctionType(newFunctionType);
  newFuncOp.setSymName(llvm::formatv("{0}_secret", callee.getSymName()).str());

  auto newCallOp = func::CallOp::create(rewriter, op.getLoc(), outputTypes,
                                        newFuncOp.getName(), inputs);
  rewriter.replaceOp(op, newCallOp);
  rewriter.eraseOp(callee);
  return newCallOp.getOperation();
}

void addContextAwareStructuralConversionPatterns(
    TypeConverter& typeConverter, RewritePatternSet& patterns,
    ConversionTarget& target, const std::string& attrName,
    CustomTypeConverter customTypeConverter) {
  patterns.add<ContextAwareFuncConversion>(typeConverter, patterns.getContext(),
                                           attrName, customTypeConverter);

  patterns.add<ConvertAny<func::ReturnOp>>(typeConverter,
                                           patterns.getContext());

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    SmallVector<Type, 4> convertedArgTypes, convertedResultTypes;
    bool changed = convertFuncSignatureUsingAttrs(
        cast<FunctionOpInterface>(op.getOperation()), convertedArgTypes,
        convertedResultTypes, attrName, customTypeConverter);
    return !changed;
  });

  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
}

}  // namespace heir
}  // namespace mlir
