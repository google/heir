#include "lib/Utils/ContextAwareConversionUtils.h"

#include <iterator>
#include <memory>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
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

#define DEBUG_TYPE "context-aware-conversion-utils"

namespace mlir {
namespace heir {

FailureOr<Operation *> convertGeneral(
    const ContextAwareTypeConverter *typeConverter, Operation *op,
    ArrayRef<Value> operands, ContextAwareConversionPatternRewriter &rewriter) {
  LLVM_DEBUG({
    llvm::dbgs() << "convertGeneral: operand types are\n\n";
    for (auto operand : operands) {
      llvm::dbgs() << "  - " << operand.getType() << "\n";
    }
    llvm::dbgs() << "\n";
  });
  SmallVector<Type> newResultTypes;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), op->getResults(),
                                         newResultTypes))) {
    LLVM_DEBUG(llvm::dbgs()
               << "convertGeneral: failed to convert result types\n");
    return failure();
  }

  SmallVector<std::unique_ptr<Region>, 1> regions;
  for (auto &region : op->getRegions()) {
    Region *newRegion = new Region(op);
    if (failed(rewriter.convertRegionTypes(&region, *typeConverter)))
      return failure();
    newRegion->takeBody(region);
    regions.push_back(std::unique_ptr<Region>(newRegion));
  }

  Operation *newOp = rewriter.create(OperationState(
      op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
      op->getAttrs(), op->getSuccessors(), regions));

  rewriter.replaceOp(op, newOp);
  return newOp;
}

static LogicalResult convertFuncOpTypes(
    FunctionOpInterface funcOp, const ContextAwareTypeConverter &typeConverter,
    ContextAwareConversionPatternRewriter &rewriter) {
  FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!type) return failure();

  // Convert the original function types.
  // HEIR: use custom callback because a func need not have a body, and/or
  // no SSA values to use as the context hook. Punt to the type converter.
  SmallVector<Type, 1> newArgTypes;
  SmallVector<Type, 1> newResultTypes;

  if (failed(typeConverter.convertFuncSignature(funcOp, newArgTypes,
                                                newResultTypes)) ||
      // HEIR: It's OK to provide a nullptr SignatureConversion to
      // convertRegionTypes.
      failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                         typeConverter,
                                         /*entryConversion=*/nullptr)))
    return failure();

  // Update the function signature in-place.
  auto newType =
      FunctionType::get(rewriter.getContext(), newArgTypes, newResultTypes);

  rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });
  LLVM_DEBUG(llvm::dbgs() << "converted function signature: " << newType
                          << "\n");

  return success();
}

LogicalResult ContextAwareFuncConversion::matchAndRewrite(
    func::FuncOp op, OpAdaptor adaptor,
    ContextAwareConversionPatternRewriter &rewriter) const {
  SmallVector<Type> oldFuncOperandTypes(op.getFunctionType().getInputs());
  SmallVector<Type> oldFuncResultTypes(op.getFunctionType().getResults());

  if (failed(convertFuncOpTypes(op, *contextAwareTypeConverter, rewriter)))
    return failure();

  if (failed(finalizeFuncOpModification(op, oldFuncOperandTypes,
                                        oldFuncResultTypes, rewriter)))
    return failure();

  return success();
}

FailureOr<Operation *> SecretGenericFuncCallConversion::matchAndRewriteInner(
    secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
    ArrayRef<NamedAttribute> attributes,
    ContextAwareConversionPatternRewriter &rewriter) const {
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

void addStructuralConversionPatterns(ContextAwareTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target) {
  patterns
      .add<ContextAwareFuncConversion, ConvertAnyContextAware<func::ReturnOp>>(
          typeConverter, patterns.getContext());

  target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op); });

  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
}

}  // namespace heir
}  // namespace mlir
