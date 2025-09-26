#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Secret/Conversions/Patterns.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
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

#define DEBUG_TYPE "secret-to-ckks"

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOCKKS
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h.inc"

namespace {

// Returns an RLWE ring given the specified number of bits needed and polynomial
// modulus degree.
FailureOr<polynomial::RingAttr> getRlweRNSRing(
    MLIRContext* ctx, const std::vector<int64_t>& primes, int polyModDegree) {
  // monomial
  std::vector<polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, polyModDegree);
  monomials.emplace_back(1, 0);
  auto result = polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  polynomial::IntPolynomial xnPlusOne = result.value();

  // moduli chain
  SmallVector<Type, 4> modTypes;
  for (auto prime : primes) {
    auto type = IntegerType::get(ctx, 64);
    modTypes.push_back(
        mod_arith::ModArithType::get(ctx, IntegerAttr::get(type, prime)));
  }

  // types
  auto rnsType = rns::RNSType::get(ctx, modTypes);
  return polynomial::RingAttr::get(
      rnsType, polynomial::IntPolynomialAttr::get(ctx, xnPlusOne));
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());

  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(level + 1));
  return polynomial::RingAttr::get(newRnsType, ringAttr.getPolynomialModulus());
}

// Returns the unique non-unit dimension of a tensor and its rank.
// Returns failure if the tensor has more than one non-unit dimension.
FailureOr<std::pair<unsigned, int64_t>> getNonUnitDimension(
    RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();

  if (llvm::count_if(shape, [](auto dim) { return dim != 1; }) != 1) {
    return failure();
  }

  unsigned nonUnitIndex = std::distance(
      shape.begin(), llvm::find_if(shape, [&](auto dim) { return dim != 1; }));

  return std::make_pair(nonUnitIndex, shape[nonUnitIndex]);
}

}  // namespace

class SecretToCKKSTypeConverter : public TypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext* ctx, polynomial::RingAttr rlweRing,
                            bool packTensorInSlots)
      : ring_(rlweRing), packTensorInSlots_(packTensorInSlots) {
    addConversion([](Type type) { return type; });
    addConversion([this](Value value) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Converting type for value " << value << "\n");
      FailureOr<Attribute> attr = findAttributeAssociatedWith(
          value, mgmt::MgmtDialect::kArgMgmtAttrName);
      if (failed(attr)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to find context attribute for " << value);
        return std::nullopt;
      }
      LLVM_DEBUG(llvm::dbgs() << "found attribute " << attr.value() << "\n");
      return convertTypeWithAttr(value.getType(), attr.value());
    });
  }

  std::optional<Type> convertTypeWithAttr(Type type, Attribute attr) const {
    auto secretType = dyn_cast<secret::SecretType>(type);
    auto mgmtAttr = dyn_cast<mgmt::MgmtAttr>(attr);
    if (secretType && mgmtAttr)
      return convertSecretTypeWithMgmtAttr(secretType, mgmtAttr);
    LLVM_DEBUG(llvm::dbgs() << "Only supported secret types with mgmt attr");
    return std::nullopt;
  }

  Type convertSecretTypeWithMgmtAttr(secret::SecretType type,
                                     mgmt::MgmtAttr mgmtAttr) const {
    auto* ctx = type.getContext();
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();
    auto scale = mgmtAttr.getScale();

    Type valueTy = type.getValueType();

    // Note that slot number for CKKS is always half of the ring dimension.
    // so ring_.getPolynomialModulus() is not useful here
    // TODO(#1191): use packing information to get the correct slot number
    auto plaintextRing = polynomial::RingAttr::get(
        type.getContext(), Float64Type::get(ctx), ring_.getPolynomialModulus());

    SmallVector<IntegerAttr, 6> moduliChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring_.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      moduliChain.push_back(modulus);
    }

    auto ciphertext = lwe::LWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, valueTy,
                                      lwe::NoOverflowAttr::get(ctx)),
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing,
            lwe::InverseCanonicalEncodingAttr::get(ctx, scale)),
        lwe::CiphertextSpaceAttr::get(ctx,
                                      getRlweRNSRingWithLevel(ring_, level),
                                      lwe::LweEncryptionType::mix, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, moduliChain, level));

    // Return a single ciphertext if the input is a scalar.
    if (!isa<TensorType>(valueTy)) return ciphertext;

    // The input is a tensor type.
    auto tensorTy = cast<RankedTensorType>(valueTy);
    // If the input is packed into a ciphertext SIMD slots (i.e. it is a tensor
    // of shape NxciphertextSize) then return Nxciphertext.
    if (this->packTensorInSlots_) {
      Type underlyingTy;
      if (tensorTy.getRank() == 1) {
        // A 1xciphertextSize tensor is packed into a single ciphertext.
        underlyingTy = valueTy;
        return ciphertext = lwe::LWECiphertextType::get(
                   ctx,
                   lwe::ApplicationDataAttr::get(ctx, underlyingTy,
                                                 lwe::NoOverflowAttr::get(ctx)),
                   lwe::PlaintextSpaceAttr::get(
                       ctx, plaintextRing,
                       lwe::InverseCanonicalEncodingAttr::get(ctx, scale)),
                   lwe::CiphertextSpaceAttr::get(
                       ctx, getRlweRNSRingWithLevel(ring_, level),
                       lwe::LweEncryptionType::mix, dimension),
                   lwe::KeyAttr::get(ctx, 0),
                   lwe::ModulusChainAttr::get(ctx, moduliChain, level));
      }
      // An NxCiphertextSize tensor is packed into N ciphertexts.
      assert(tensorTy.getRank() == 2 && "expected rank 1 or 2 tensor");
      underlyingTy = RankedTensorType::get(tensorTy.getShape().drop_front(),
                                           tensorTy.getElementType());
      auto ciphertext = lwe::LWECiphertextType::get(
          ctx,
          lwe::ApplicationDataAttr::get(ctx, underlyingTy,
                                        lwe::NoOverflowAttr::get(ctx)),
          lwe::PlaintextSpaceAttr::get(
              ctx, plaintextRing,
              lwe::InverseCanonicalEncodingAttr::get(ctx, scale)),
          lwe::CiphertextSpaceAttr::get(ctx,
                                        getRlweRNSRingWithLevel(ring_, level),
                                        lwe::LweEncryptionType::mix, dimension),
          lwe::KeyAttr::get(ctx, 0),
          lwe::ModulusChainAttr::get(ctx, moduliChain, level));
      return RankedTensorType::get(tensorTy.getShape().drop_back(), ciphertext);
    }
    // If the input IR does not contain aligned ciphertexts, we will not
    // pack tensors into ciphertext SIMD slots, so tensors are converted
    // into tensors of RLWE ciphertexts.
    ciphertext = lwe::LWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, getElementTypeOrSelf(valueTy),
                                      lwe::NoOverflowAttr::get(ctx)),
        ciphertext.getPlaintextSpace(), ciphertext.getCiphertextSpace(),
        ciphertext.getKey(), ciphertext.getModulusChain());
    return RankedTensorType::get(tensorTy.getShape(), ciphertext);
  }

 private:
  polynomial::RingAttr ring_;
  bool packTensorInSlots_;
};

class SecretGenericTensorExtractConversion
    : public SecretGenericOpConversion<tensor::ExtractOp, ckks::ExtractOp> {
 public:
  using SecretGenericOpConversion<tensor::ExtractOp,
                                  ckks::ExtractOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    auto inputTy = inputs[0].getType();
    if (!isa<lwe::LWECiphertextType>(getElementTypeOrSelf(inputTy))) {
      return failure();
    }
    if (isa<RankedTensorType>(inputTy)) {
      // TODO(#1174): decide this in earlier pipeline
      // Extracts an element out of a tensor (the secret tensor is not
      // packed).
      return rewriter
          .replaceOpWithNewOp<tensor::ExtractOp>(op, outputTypes, inputs)
          .getOperation();
    }
    // Extracts an element out of a slot of a single ciphertext.
    // TODO(#913): Once we have a layout descriptor, we should be able to
    // translate a tensor.extract into the appropriate ckks.extract operation.
    // For now, if there we are extracting a multi-dimensional tensor with
    // only one non-unit dimension stored in a single ciphertext along that
    // dimension, then extract on the index of the non-unit dimension.
    auto lweCiphertextInputTy = cast<lwe::LWECiphertextType>(inputTy);
    auto underlyingTy = cast<RankedTensorType>(
        lweCiphertextInputTy.getApplicationData().getMessageType());
    auto nonUnitDim = getNonUnitDimension(underlyingTy);
    if (failed(nonUnitDim)) {
      return failure();
    }
    assert(inputs.size() == 1 + underlyingTy.getRank() &&
           "expected tensor.extract inputs for each index");
    auto nonUnitShift = inputs[1 + nonUnitDim.value().first];
    return rewriter
        .replaceOpWithNewOp<ckks::ExtractOp>(op, outputTypes[0], inputs[0],
                                             nonUnitShift)
        .getOperation();
  }
};

class SecretGenericTensorInsertConversion
    : public SecretGenericOpConversion<tensor::InsertOp, tensor::InsertOp> {
 public:
  using SecretGenericOpConversion<tensor::InsertOp,
                                  tensor::InsertOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    if (!isa<lwe::LWECiphertextType>(inputs[0].getType())) {
      op.emitError()
          << "expected scalar to insert to be of type RLWE ciphertext"
          << inputs[0].getType();
      return failure();
    }
    if (isa<RankedTensorType>(inputs[1].getType())) {
      // Insert an element into a tensor (the secret tensor is not packed).
      return rewriter
          .replaceOpWithNewOp<tensor::InsertOp>(op, outputTypes, inputs)
          .getOperation();
    }
    // We can also support the case where the secret tensor is packed into a
    // single ciphertext by converting the insert operation into a zero-hot
    // multiplication followed by an addition of the scalar encoded into a
    // plaintext in the correct slot.
    return failure();
  }
};

class SecretGenericTensorExpandConversion
    : public SecretGenericOpConversion<tensor::ExpandShapeOp,
                                       tensor::ExpandShapeOp> {
 public:
  using SecretGenericOpConversion<
      tensor::ExpandShapeOp, tensor::ExpandShapeOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    // We expect this operation to occur when dropping unit dimensions in order
    // to allow rotation ops to operate on 1-D tensors.
    SliceVerificationResult res = isRankReducedType(
        cast<ShapedType>(
            cast<secret::SecretType>(op.getResultTypes()[0]).getValueType()),
        cast<ShapedType>(
            cast<secret::SecretType>(op.getOperandTypes()[0]).getValueType()));
    if (res != SliceVerificationResult::Success) {
      return rewriter.notifyMatchFailure(
          op, "expected input type to be a rank reduced type of the output");
    }
    if (!isa<lwe::LWECiphertextType>(inputs[0].getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected input that was expanded to be of type RLWE ciphertext");
    }

    if (!isa<RankedTensorType>(outputTypes[0])) {
      return rewriter.notifyMatchFailure(
          op, "expected expanded output to be a ranked tensor");
    }
    return rewriter
        .replaceOpWithNewOp<tensor::FromElementsOp>(op, outputTypes, inputs)
        .getOperation();
  }
};

class SecretGenericTensorCollapseConversion
    : public SecretGenericOpConversion<tensor::CollapseShapeOp,
                                       tensor::CollapseShapeOp> {
 public:
  using SecretGenericOpConversion<
      tensor::CollapseShapeOp,
      tensor::CollapseShapeOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    // We expect this operation to occur when dropping unit dimensions in order
    // to allow rotation ops to operate on 1-D tensors.
    SliceVerificationResult res = isRankReducedType(
        cast<ShapedType>(
            cast<secret::SecretType>(op.getOperandTypes()[0]).getValueType()),
        cast<ShapedType>(
            cast<secret::SecretType>(op.getResultTypes()[0]).getValueType()));
    if (res != SliceVerificationResult::Success) {
      return rewriter.notifyMatchFailure(
          op, "expected input type to be a rank reduced type of the output");
    }
    if (!isa<RankedTensorType>(inputs[0].getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected input that was collapsed to be a ranked tensor");
    }
    if (!isa<lwe::LWECiphertextType>(outputTypes[0])) {
      return rewriter.notifyMatchFailure(
          op, "expected collapsed output to be of type RLWE ciphertext");
    }

    Value idx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    return rewriter
        .replaceOpWithNewOp<tensor::ExtractOp>(op, outputTypes[0], inputs[0],
                                               idx)
        .getOperation();
  }
};

bool hasSecretOperandsOrResults(Operation* op) {
  return llvm::any_of(op->getOperands(),
                      [](Value operand) {
                        return isa<secret::SecretType>(operand.getType());
                      }) ||
         llvm::any_of(op->getResults(), [](Value result) {
           return isa<secret::SecretType>(result.getType());
         });
}

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

    // NOTE: 2 ** logN != polyModDegree
    // they have different semantic
    // auto logN = schemeParamAttr.getLogN();

    // pass option polyModDegree is actually the number of slots
    // TODO(#1402): use a proper name for CKKS
    auto rlweRing = getRlweRNSRing(context, schemeParamAttr.getQ().asArrayRef(),
                                   polyModDegree);
    if (failed(rlweRing)) {
      return signalPassFailure();
    }

    bool usePublicKey =
        schemeParamAttr.getEncryptionType() == ckks::CKKSEncryptionType::pk;

    // Ensure that all secret types are uniform and matching the ring
    // parameter size in order to pack tensors into ciphertext SIMD slots.
    // TODO(#1174): decide this earlier, remove polyModDegree param to earlier
    // pipeline
    LogicalResult validationResult =
        walkAndValidateValues(module, [&](Value value) {
          if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
            auto tensorTy = dyn_cast<RankedTensorType>(secretTy.getValueType());
            if (tensorTy) {
              // TODO(#913): Multidimensional tensors with a single non-unit
              // dimension are assumed to be packed in the order of that
              // dimensions.
              auto nonUnitDim = getNonUnitDimension(tensorTy);
              if (failed(nonUnitDim) ||
                  nonUnitDim.value().second != polyModDegree) {
                return failure();
              }
            }
          }
          return success();
        });
    if (failed(validationResult)) {
      emitWarning(module->getLoc(),
                  "expected secret types to be tensors with dimension matching "
                  "ring parameter, pass will not pack tensors into ciphertext "
                  "SIMD slots");
    }
    bool packTensorInSlots = succeeded(validationResult);

    // Invariant: for every SecretType, there is a
    // corresponding MgmtAttr attached to it,
    // either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp).
    // Otherwise the typeConverter won't find the proper type information
    // and fail
    SecretToCKKSTypeConverter typeConverter(context, rlweRing.value(),
                                            packTensorInSlots);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<mgmt::ModReduceOp, mgmt::RelinearizeOp>();
    // for mod reduce on tensor ciphertext
    target.addLegalOp<arith::ConstantOp, tensor::EmptyOp>();

    target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineYieldOp>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<tensor::ExtractOp, tensor::ExtractSliceOp,
                                 tensor::InsertOp, tensor::ExpandShapeOp,
                                 tensor::CollapseShapeOp>(
        [&](Operation* op) { return typeConverter.isLegal(op); });

    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return !hasSecretOperandsOrResults(op); });

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
        SecretGenericOpConversion<arith::ExtSIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::ExtUIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::FPToSIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::FPToUIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::SIToFPOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::UIToFPOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::MulFOp, ckks::MulOp>,
        SecretGenericOpConversion<arith::MulIOp, ckks::MulOp>,
        SecretGenericOpConversion<arith::SubFOp, ckks::SubOp>,
        SecretGenericOpConversion<arith::SubIOp, ckks::SubOp>,
        SecretGenericOpConversion<mgmt::BootstrapOp, ckks::BootstrapOp>,
        SecretGenericOpConversion<tensor::EmptyOp, tensor::EmptyOp>,
        SecretGenericOpModulusSwitchConversion<ckks::RescaleOp>,
        SecretGenericOpRelinearizeConversion<ckks::RelinearizeOp>,
        SecretGenericOpRotateConversion<ckks::RotateOp>,
        SecretGenericOpLevelReduceConversion<ckks::LevelReduceOp>,
        SecretGenericTensorExtractConversion,
        SecretGenericTensorInsertConversion,
        SecretGenericTensorCollapseConversion,
        SecretGenericTensorExpandConversion, ConvertAny<affine::AffineForOp>,
        ConvertAny<affine::AffineYieldOp>, ConvertAny<tensor::ExtractSliceOp>,
        ConvertAny<tensor::ExtractOp>, ConvertAny<tensor::InsertOp>,
        SecretGenericFuncCallConversion>(typeConverter, context);

    patterns.add<ConvertClientConceal>(typeConverter, context, usePublicKey,
                                       rlweRing.value());
    patterns.add<ConvertClientReveal>(typeConverter, context, rlweRing.value());

    addContextAwareStructuralConversionPatterns(
        typeConverter, patterns, target,
        std::string(mgmt::MgmtDialect::kArgMgmtAttrName),
        [&](Type type, Attribute attr) {
          return typeConverter.convertTypeWithAttr(type, attr);
        });

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }

    clearAttrs(getOperation(), mgmt::MgmtDialect::kArgMgmtAttrName);
    mgmt::cleanupInitOp(getOperation());
  }
};

}  // namespace mlir::heir
