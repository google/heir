#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"

#include <cassert>
#include <cstdint>
#include <iterator>
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
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
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

namespace {

// Returns an RLWE ring given the specified number of bits needed and polynomial
// modulus degree.
FailureOr<polynomial::RingAttr> getRlweRNSRing(
    MLIRContext *ctx, const std::vector<int64_t> &primes, int polyModDegree) {
  // monomial
  std::vector<::mlir::heir::polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, polyModDegree);
  monomials.emplace_back(1, 0);
  auto result =
      ::mlir::heir::polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  ::mlir::heir::polynomial::IntPolynomial xnPlusOne = result.value();

  // moduli chain
  SmallVector<Type, 4> modTypes;
  for (auto prime : primes) {
    auto type = IntegerType::get(ctx, 64);
    modTypes.push_back(
        mod_arith::ModArithType::get(ctx, IntegerAttr::get(type, prime)));
  }

  // types
  auto rnsType = rns::RNSType::get(ctx, modTypes);
  return ::mlir::heir::polynomial::RingAttr::get(
      rnsType, polynomial::IntPolynomialAttr::get(ctx, xnPlusOne));
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());

  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(level + 1));
  return ::mlir::heir::polynomial::RingAttr::get(
      newRnsType, ringAttr.getPolynomialModulus());
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

class SecretToCKKSTypeConverter
    : public UniquelyNamedAttributeAwareTypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext *ctx,
                            ::mlir::heir::polynomial::RingAttr rlweRing,
                            bool packTensorInSlots)
      : UniquelyNamedAttributeAwareTypeConverter(
            mgmt::MgmtDialect::kArgMgmtAttrName) {
    addConversion([](Type type, Attribute attr) { return type; });
    addConversion([this](secret::SecretType type, mgmt::MgmtAttr mgmtAttr) {
      return convertSecretTypeWithMgmtAttr(type, mgmtAttr);
    });

    ring_ = rlweRing;
    packTensorInSlots_ = packTensorInSlots;
  }

  Type convertSecretTypeWithMgmtAttr(secret::SecretType type,
                                     mgmt::MgmtAttr mgmtAttr) const {
    auto *ctx = type.getContext();
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();
    auto scale = mgmtAttr.getScale();

    Type valueTy = type.getValueType();

    // Note that slot number for CKKS is always half of the ring dimension.
    // so ring_.getPolynomialModulus() is not useful here
    // TODO(#1191): use packing information to get the correct slot number
    auto plaintextRing = ::mlir::heir::polynomial::RingAttr::get(
        type.getContext(), Float64Type::get(ctx), ring_.getPolynomialModulus());

    SmallVector<IntegerAttr, 6> moduliChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring_.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      moduliChain.push_back(modulus);
    }

    auto ciphertext = lwe::NewLWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, type.getValueType(),
                                      lwe::NoOverflowAttr::get(ctx)),
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing,
            lwe::InverseCanonicalEncodingAttr::get(ctx, scale)),
        lwe::CiphertextSpaceAttr::get(ctx,
                                      getRlweRNSRingWithLevel(ring_, level),
                                      lwe::LweEncryptionType::mix, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, moduliChain, level));

    // Return a single ciphertext if inputs are packed into a single
    // ciphertext SIMD slot or the secret value type is a scalar.
    if (this->packTensorInSlots_ || !isa<TensorType>(valueTy)) {
      return ciphertext;
    }
    // If the input IR does not contain aligned ciphertexts, we will not
    // pack tensors into ciphertext SIMD slots, so tensors are converted
    // into tensors of RLWE ciphertexts.
    assert(dyn_cast<RankedTensorType>(valueTy) &&
           "expected ranked tensor type");
    auto scalarType = cast<RankedTensorType>(valueTy).getElementType();
    ciphertext = lwe::NewLWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, scalarType,
                                      lwe::NoOverflowAttr::get(ctx)),
        ciphertext.getPlaintextSpace(), ciphertext.getCiphertextSpace(),
        ciphertext.getKey(), ciphertext.getModulusChain());
    return RankedTensorType::get(cast<RankedTensorType>(valueTy).getShape(),
                                 ciphertext);
  }

 private:
  ::mlir::heir::polynomial::RingAttr ring_;
  bool packTensorInSlots_;
};

class SecretGenericTensorExtractConversion
    : public SecretGenericOpConversion<tensor::ExtractOp, ckks::ExtractOp> {
 public:
  using SecretGenericOpConversion<tensor::ExtractOp,
                                  ckks::ExtractOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    auto inputTy = inputs[0].getType();
    if (!isa<lwe::NewLWECiphertextType>(getElementTypeOrSelf(inputTy))) {
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
    auto lweCiphertextInputTy = cast<lwe::NewLWECiphertextType>(inputTy);
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

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    if (!isa<lwe::NewLWECiphertextType>(inputs[0].getType())) {
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

bool hasSecretOperandsOrResults(Operation *op) {
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
    MLIRContext *context = &getContext();
    auto *module = getOperation();

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
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<tensor::ExtractOp, tensor::ExtractSliceOp,
                                 tensor::InsertOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return !hasSecretOperandsOrResults(op); });

    patterns.add<
        SecretGenericOpCipherPlainConversion<arith::AddFOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulFOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubFOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, ckks::SubPlainOp>,
        SecretGenericOpConversion<arith::AddFOp, ckks::AddOp>,
        SecretGenericOpConversion<arith::AddIOp, ckks::AddOp>,
        SecretGenericOpConversion<arith::ExtSIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::ExtUIOp,
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
        ConvertAnyContextAware<affine::AffineForOp>,
        ConvertAnyContextAware<affine::AffineYieldOp>,
        ConvertAnyContextAware<tensor::ExtractSliceOp>,
        ConvertAnyContextAware<tensor::ExtractOp>,
        ConvertAnyContextAware<tensor::InsertOp>,
        SecretGenericFuncCallConversion>(typeConverter, context);

    patterns.add<ConvertClientConceal>(typeConverter, context, usePublicKey,
                                       rlweRing.value());
    patterns.add<ConvertClientReveal>(typeConverter, context, rlweRing.value());

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    clearAttrs(getOperation(), mgmt::MgmtDialect::kArgMgmtAttrName);
  }
};

}  // namespace mlir::heir
