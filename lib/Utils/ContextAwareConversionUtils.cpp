#include "lib/Utils/ContextAwareConversionUtils.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "context-aware-conversion-utils"

namespace mlir {
namespace heir {

FailureOr<Value> encodeCleartextAsPlaintext(
    ImplicitLocOpBuilder& builder, Value cleartext,
    lwe::LWECiphertextType ciphertextElementType, mgmt::MgmtAttr mgmtAttr,
    llvm::raw_string_ostream& errStr) {
  MLIRContext* ctx = builder.getContext();
  lwe::PlaintextSpaceAttr plaintextSpace =
      ciphertextElementType.getPlaintextSpace();
  Attribute ciphertextEncoding = plaintextSpace.getEncoding();
  Attribute plaintextEncoding = lwe::getEncodingAttrWithNewScalingFactor(
      ciphertextEncoding, mgmtAttr.getScale());

  if (!plaintextEncoding) {
    errStr << "failed to compute plaintext encoding";
    return failure();
  }

  // TODO(#1643): inherit level information to plaintext type from init-op
  // mgmt attr. This actually needs to make LWEPlaintextType RNS aware.
  auto plaintextTy = lwe::LWEPlaintextType::get(
      ctx, lwe::PlaintextSpaceAttr::get(ctx, plaintextSpace.getRing(),
                                        plaintextEncoding));

  // cleartext is a ciphertext-semantic tensor, so it
  // could be a tensor<Nxty>, tensor<k x N x ty>, (where k=1 is possible).
  // For rank 1, it's a single ciphertext and can be encoded directly.
  auto cleartextTensorTy = cast<RankedTensorType>(cleartext.getType());
  int64_t numSlots =
      cleartextTensorTy.getDimSize(cleartextTensorTy.getRank() - 1);
  if (cleartextTensorTy.getRank() == 1) {
    Value encodeOp =
        lwe::RLWEEncodeOp::create(builder, plaintextTy, cleartext,
                                  plaintextEncoding, plaintextSpace.getRing())
            .getResult();
    return encodeOp;
  }

  assert(cleartextTensorTy.getRank() == 2);
  // For higher rank, we need to extract all the inner rank-1 tensors, encode
  // them, and reassemble.
  auto sliceTy =
      RankedTensorType::get({numSlots}, cleartextTensorTy.getElementType());
  SmallVector<Value> encodedSlices;
  for (int64_t i = 0; i < cleartextTensorTy.getShape()[0]; ++i) {
    SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(i),
                                         builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(numSlots)};
    SmallVector<OpFoldResult> strides(2, builder.getIndexAttr(1));
    auto slice = tensor::ExtractSliceOp::create(builder, sliceTy, cleartext,
                                                offsets, sizes, strides);
    Value encodedSlice =
        lwe::RLWEEncodeOp::create(builder, plaintextTy, slice,
                                  plaintextEncoding, plaintextSpace.getRing())
            .getResult();
    encodedSlices.push_back(encodedSlice);
  }

  auto reassembledEncodedSlices = tensor::FromElementsOp::create(
      builder,
      RankedTensorType::get({cleartextTensorTy.getShape()[0]}, plaintextTy),
      encodedSlices);
  return reassembledEncodedSlices.getResult();
}

FailureOr<Operation*> convertGeneral(
    const ContextAwareTypeConverter* typeConverter, Operation* op,
    ArrayRef<Value> operands, ContextAwareConversionPatternRewriter& rewriter) {
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
  for (auto& region : op->getRegions()) {
    Region* newRegion = new Region(op);
    if (failed(rewriter.convertRegionTypes(&region, *typeConverter)))
      return failure();
    newRegion->takeBody(region);
    regions.push_back(std::unique_ptr<Region>(newRegion));
  }

  Operation* newOp = rewriter.create(OperationState(
      op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
      op->getAttrs(), op->getSuccessors(), regions));

  rewriter.replaceOp(op, newOp);
  return newOp;
}

static LogicalResult convertFuncOpTypes(
    FunctionOpInterface funcOp, const ContextAwareTypeConverter& typeConverter,
    ContextAwareConversionPatternRewriter& rewriter) {
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
    ContextAwareConversionPatternRewriter& rewriter) const {
  SmallVector<Type> oldFuncOperandTypes(op.getFunctionType().getInputs());
  SmallVector<Type> oldFuncResultTypes(op.getFunctionType().getResults());

  if (failed(convertFuncOpTypes(op, *contextAwareTypeConverter, rewriter)))
    return failure();

  if (failed(finalizeFuncOpModification(op, oldFuncOperandTypes,
                                        oldFuncResultTypes, rewriter)))
    return failure();

  return success();
}

FailureOr<Operation*> SecretGenericFuncCallConversion::matchAndRewriteInner(
    secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
    ArrayRef<NamedAttribute> attributes,
    ContextAwareConversionPatternRewriter& rewriter) const {
  // check if any args are secret from wrapping generic
  // clone the callee (and update a unique name, for now always) the call
  // operands add a note that we don't have to always clone to be secret
  // update the called func's type signature

  func::CallOp callOp = *op.getBody()->getOps<func::CallOp>().begin();
  auto maybeCallee = getCalledFunction(callOp);
  if (failed(maybeCallee)) {
    return op->emitError() << "failed to get called function";
  }
  auto callee = maybeCallee.value();

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

void addStructuralConversionPatterns(ContextAwareTypeConverter& typeConverter,
                                     RewritePatternSet& patterns,
                                     ConversionTarget& target) {
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
