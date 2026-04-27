#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DEF_CKKSTOLWE
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h.inc"

template <typename OriginalOp, typename NewOp>
struct ConvertOp : public OpConversionPattern<OriginalOp> {
  using OpConversionPattern<OriginalOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OriginalOp op, typename OriginalOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    OperationState state(op.getLoc(), NewOp::getOperationName());
    state.addOperands(adaptor.getOperands());
    state.addAttributes(op->getAttrs());
    state.addTypes(op->getResultTypes());
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertKeySwitchInner : public OpConversionPattern<KeySwitchInnerOp> {
  using OpConversionPattern<KeySwitchInnerOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      KeySwitchInnerOp op, KeySwitchInnerOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    SchemeParamAttr schemeParamAttr =
        op->getParentOfType<ModuleOp>()->getAttrOfType<SchemeParamAttr>(
            CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      return op->emitOpError()
             << "Cannot find scheme param attribute on parent module";
    }
    auto schemeParam = getSchemeParamFromAttr(schemeParamAttr);

    int64_t partSize = schemeParam.getPi().size();
    if (!partSize) {
      return rewriter.notifyMatchFailure(
          op, "Cannot lower key_switch_inner with empty modulus chain");
    }

    auto ringEltType = cast<lwe::LWERingEltType>(op.getValue().getType());
    auto inputRNSType =
        dyn_cast<rns::RNSType>(ringEltType.getRing().getCoefficientType());
    if (!inputRNSType) {
      return rewriter.notifyMatchFailure(
          op, "Input type must be an RNS ring element");
    }

    int rnsLength = inputRNSType.getBasisTypes().size();
    int64_t numFullPartitions = rnsLength / partSize;
    int64_t extraPartStart = partSize * numFullPartitions;
    int64_t extraPartSize = rnsLength - extraPartStart;
    SmallVector<Value> partitions;
    for (int i = 0; i < numFullPartitions; ++i) {
      partitions.push_back(lwe::ExtractSliceOp::create(
          b, adaptor.getValue(), b.getIndexAttr(i * partSize),
          b.getIndexAttr(partSize)));
    }
    if (extraPartSize > 0) {
      partitions.push_back(lwe::ExtractSliceOp::create(
          b, adaptor.getValue(), b.getIndexAttr(extraPartStart),
          b.getIndexAttr(extraPartSize)));
    }

    SmallVector<Type> extModuli;
    for (auto ty : inputRNSType.getBasisTypes()) {
      extModuli.push_back(ty);
    }
    for (auto prime : schemeParam.getPi()) {
      extModuli.push_back(mod_arith::ModArithType::get(
          op.getContext(), b.getI64IntegerAttr(prime)));
    }
    rns::RNSType newBasisType = rns::RNSType::get(op.getContext(), extModuli);

    SmallVector<Value> extendedPartitions;
    for (auto value : partitions) {
      extendedPartitions.push_back(
          lwe::ConvertBasisOp::create(b, value, TypeAttr::get(newBasisType)));
    }

    int k = extendedPartitions.size();
    Value sum;
    for (int i = 0; i < k; i++) {
      Value idx = arith::ConstantIndexOp::create(b, i).getResult();
      Value ksk =
          tensor::ExtractOp::create(b, adaptor.getKeySwitchingKey(), idx);
      Value prod = lwe::RMulRingEltOp::create(b, extendedPartitions[i], ksk);
      if (i == 0) {
        sum = prod;
      } else {
        sum = lwe::RAddOp::create(b, sum, prod);
      }
    }

    Value constTerm = lwe::ExtractCoeffOp::create(b, sum, b.getIndexAttr(0));
    Value linearTerm = lwe::ExtractCoeffOp::create(b, sum, b.getIndexAttr(1));
    Value modDownConstTerm =
        lwe::ConvertBasisOp::create(b, constTerm, TypeAttr::get(inputRNSType));
    Value modDownLinearTerm =
        lwe::ConvertBasisOp::create(b, linearTerm, TypeAttr::get(inputRNSType));

    rewriter.replaceOp(op, {modDownConstTerm, modDownLinearTerm});
    return success();
  }
};

struct CKKSToLWE : public impl::CKKSToLWEBase<CKKSToLWE> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, ckks::CKKSDialect,
                           lwe::LWEDialect, tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<AddOp, AddPlainOp, SubOp, SubPlainOp, NegateOp, MulOp,
                        MulPlainOp, KeySwitchInnerOp>();

    RewritePatternSet patterns(context);
    patterns.add<
        ConvertKeySwitchInner, ConvertOp<AddOp, lwe::RAddOp>,
        ConvertOp<AddPlainOp, lwe::RAddPlainOp>, ConvertOp<SubOp, lwe::RSubOp>,
        ConvertOp<SubPlainOp, lwe::RSubPlainOp>,
        ConvertOp<NegateOp, lwe::RNegateOp>, ConvertOp<MulOp, lwe::RMulOp>,
        ConvertOp<MulPlainOp, lwe::RMulPlainOp>>(context);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyFullConversion(module, target, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::ckks
