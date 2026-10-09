#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Secret/Conversions/Patterns.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/LWE/IR/LWEDialect.h"
// IWYU pragma: end_keep

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOCKKS
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h.inc"

class SecretToCKKSTypeConverter : public SecretToRlweTypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext* ctx, polynomial::RingAttr rlweRing)
      : SecretToRlweTypeConverter(ctx, rlweRing) {}

 protected:
  polynomial::RingAttr getPlaintextRing(MLIRContext* ctx) const override {
    // Note that slot number for CKKS is always half of the ring dimension.
    // so ring.getPolynomialModulus() is not useful here
    // TODO(#2764): use packing information to get the correct slot number
    return polynomial::RingAttr::get(ctx, Float64Type::get(ctx),
                                     ring.getPolynomialModulus());
  }

  Attribute getEncodingAttr(MLIRContext* ctx, int64_t scale) const override {
    return lwe::InverseCanonicalEncodingAttr::get(ctx, scale);
  }

  lwe::LweEncryptionType getEncryptionType() const override {
    return lwe::LweEncryptionType::mix;
  }
};

class SecretGenericPlaintextDivision
    : public SecretGenericOpConversion<arith::DivFOp, ckks::MulPlainOp> {
 public:
  using SecretGenericOpConversion<arith::DivFOp,
                                  ckks::MulPlainOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    // Check that the divisor is a plaintext.
    Value ciphertextInput = inputs[0];
    Value cleartextDivisor = inputs[1];
    if (isa<lwe::LWECiphertextType>(cleartextDivisor.getType()) ||
        !isa<lwe::LWECiphertextType>(ciphertextInput.getType())) {
      return op.emitOpError(
          "ciphertext division is not supported in CKKS; expected plaintext "
          "divisor and ciphertext dividend");
    }

    // Encode 1/cleartext as a plaintext
    auto initOp =
        dyn_cast_or_null<mgmt::InitOp>(cleartextDivisor.getDefiningOp());
    if (!initOp) {
      return rewriter.notifyMatchFailure(
          op, "expected plaintext divisor to be defined by mgmt.init");
    }
    auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp);
    if (!mgmtAttr) {
      return rewriter.notifyMatchFailure(
          op, "expected plaintext divisor to have mgmt.mgmt attribute");
    }

    Value realCleartext = initOp.getInput();
    Value invertedCleartext = arith::DivFOp::create(
        rewriter, op.getLoc(),
        arith::ConstantOp::create(rewriter, op.getLoc(),
                                  realCleartext.getType(),
                                  rewriter.getOneAttr(realCleartext.getType())),
        realCleartext);

    std::string errMsg;
    llvm::raw_string_ostream errStream(errMsg);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto ciphertextElementTy = cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(ciphertextInput.getType()));
    FailureOr<Value> encodedPlaintext = encodeCleartextAsPlaintext(
        b, invertedCleartext, ciphertextElementTy, mgmtAttr, errStream);
    if (failed(encodedPlaintext)) {
      return rewriter.notifyMatchFailure(op, errMsg);
    }

    return rewriter
        .replaceOpWithNewOp<ckks::MulPlainOp>(op, ciphertextInput,
                                              encodedPlaintext.value())
        .getOperation();
  }
};

struct LinearTransformOpConversion
    : public ContextAwareOpConversionPattern<secret::GenericOp> {
  LinearTransformOpConversion(const ContextAwareTypeConverter& typeConverter_,
                              MLIRContext* context, PatternBenefit benefit = 1)
      : ContextAwareOpConversionPattern<secret::GenericOp>(typeConverter_,
                                                           context, benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    if (op.getBody()->getOperations().size() > 2) {
      return failure();
    }

    auto& innerOp = op.getBody()->getOperations().front();
    auto ltOp = dyn_cast<kernel::LinearTransformOp>(innerOp);
    if (!ltOp) {
      return failure();
    }

    // Convert inputs
    SmallVector<Value> inputs;
    for (Value operand : ltOp->getOperands()) {
      if (auto* secretArg = op.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    // Convert result types
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                op.getResults(), resultTypes)))
      return op.emitOpError(
          "failed to convert result types to CKKS ciphertext types");

    // Preserve attributes (similar to SecretGenericOpConversion)
    SmallVector<NamedAttribute> attrsToPreserve;
    for (auto& namedAttr : ltOp->getDialectAttrs()) {
      attrsToPreserve.push_back(namedAttr);
    }
    for (auto attrName : ltOp.getAttributeNames()) {
      if (auto attr = ltOp->getAttr(attrName)) {
        attrsToPreserve.push_back(rewriter.getNamedAttr(attrName, attr));
      }
    }

    // Handle mgmt attrs
    convertArrayOfDicts(op.getAllResultAttrsAttr(), attrsToPreserve);
    convertArrayOfDicts(op.getAllOperandAttrsAttr(), attrsToPreserve);
    DenseSet<StringRef> seenNames;
    SmallVector<NamedAttribute> dedupedAttrsToPreserve;
    for (auto attr : llvm::reverse(attrsToPreserve)) {
      if (seenNames.insert(attr.getName().getValue()).second) {
        dedupedAttrsToPreserve.push_back(attr);
      }
    }
    std::reverse(dedupedAttrsToPreserve.begin(), dedupedAttrsToPreserve.end());
    auto newLtOp = kernel::LinearTransformOp::create(
        rewriter, ltOp.getLoc(), resultTypes, inputs, dedupedAttrsToPreserve);

    rewriter.replaceOp(op, newLtOp->getResults());
    return success();
  }
};

struct SecretToCKKS : public impl::SecretToCKKSBase<SecretToCKKS> {
  using SecretToCKKSBase::SecretToCKKSBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      module->emitError("expected CKKS scheme parameters");
      signalPassFailure();
      return;
    }

    // NOTE: 2 ** logN != minSlotCount
    // they have different semantic
    // auto logN = schemeParamAttr.getLogN();

    // pass option minSlotCount is actually the number of slots
    // TODO(#1402): use a proper name for CKKS
    auto rlweRing =
        lwe::getRlweRNSRing(context, schemeParamAttr.getQ().asArrayRef(),
                            1 << schemeParamAttr.getLogN());
    if (failed(rlweRing)) {
      return signalPassFailure();
    }

    bool usePublicKey =
        schemeParamAttr.getEncryptionType() == ckks::CKKSEncryptionType::pk;

    // Invariant: for every SecretType, there is a
    // corresponding MgmtAttr attached to it,
    // either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp).
    // Otherwise the typeConverter won't find the proper type information
    // and fail
    SecretToCKKSTypeConverter typeConverter(context, rlweRing.value());
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addSecretToSchemeDefaultConversionTargetsAndPatterns(patterns, target,
                                                         typeConverter);

    target.addLegalDialect<ckks::CKKSDialect>();
    patterns.add<
        SecretGenericOpCipherPlainConversion<arith::AddFOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulFOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubFOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, ckks::SubPlainOp>,
        SecretGenericOpConversion<arith::NegFOp, ckks::NegateOp>,
        SecretGenericOpConversion<arith::AddFOp, ckks::AddOp>,
        SecretGenericOpConversion<arith::AddIOp, ckks::AddOp>,
        SecretGenericOpConversion<arith::MulFOp, ckks::MulOp>,
        SecretGenericOpConversion<arith::MulIOp, ckks::MulOp>,
        SecretGenericOpConversion<arith::SubFOp, ckks::SubOp>,
        SecretGenericOpConversion<arith::SubIOp, ckks::SubOp>,
        SecretGenericOpConversion<mgmt::BootstrapOp, ckks::BootstrapOp>,
        SecretGenericOpModulusSwitchConversion<ckks::RescaleOp>,
        SecretGenericOpRelinearizeConversion<ckks::RelinearizeOp>,
        SecretGenericOpRotateConversion<ckks::RotateOp>,
        SecretGenericPlaintextDivision,
        SecretGenericOpConversion<kernel::EvalChebyshevOp>,
        SecretGenericOpLevelReduceConversion<ckks::LevelReduceOp>,
        LinearTransformOpConversion>(typeConverter, context);

    patterns.add<ConvertClientConceal>(typeConverter, context, usePublicKey,
                                       rlweRing.value());
    patterns.add<ConvertClientReveal>(typeConverter, context, rlweRing.value());

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    clearAttrs(getOperation(), mgmt::MgmtDialect::kArgMgmtAttrName);
    mgmt::cleanupInitOp(getOperation());
  }
};

}  // namespace mlir::heir
