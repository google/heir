#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
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
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
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

}  // namespace

class SecretToCKKSTypeConverter
    : public UniquelyNamedAttributeAwareTypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext* ctx, polynomial::RingAttr rlweRing)
      : UniquelyNamedAttributeAwareTypeConverter(
            mgmt::MgmtDialect::kArgMgmtAttrName) {
    addConversion([](Type type, Attribute attr) { return type; });
    addConversion(
        [this](RankedTensorType type, mgmt::MgmtAttr mgmtAttr) -> Type {
          // For cases like tensor.empty + mgmt.init, we need to convert this
          // to a ciphertext type.
          //
          // Care must be taken to ensure that types that have already been
          // converted (i.e., their element type is a ciphertext or plaintext
          // type) are returned as-is, or else legality checking will consider
          // ops with these types illegal.
          if (isa<lwe::LWECiphertextType, lwe::LWEPlaintextType>(
                  type.getElementType()))
            return type;
          return convertSecretTypeWithMgmtAttr(secret::SecretType::get(type),
                                               mgmtAttr);
        });
    addConversion([this](secret::SecretType type, mgmt::MgmtAttr mgmtAttr) {
      return convertSecretTypeWithMgmtAttr(type, mgmtAttr);
    });

    ring_ = rlweRing;
  }

  Type convertSecretTypeWithMgmtAttr(secret::SecretType type,
                                     mgmt::MgmtAttr mgmtAttr) const {
    auto* ctx = type.getContext();
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();
    auto scale = mgmtAttr.getScale();
    auto tensorValueType = dyn_cast<RankedTensorType>(type.getValueType());

    // Note that slot number for CKKS is always half of the ring dimension.
    // so ring_.getPolynomialModulus() is not useful here
    // TODO(#1191): use packing information to get the correct slot number
    auto plaintextRing = polynomial::RingAttr::get(
        ctx, Float64Type::get(ctx), ring_.getPolynomialModulus());

    SmallVector<IntegerAttr, 6> modulusChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring_.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      modulusChain.push_back(modulus);
    }

    Type messageType = type.getValueType();
    if (tensorValueType && tensorValueType.getRank() > 1) {
      // The value type here is a ciphertext-semantic tensor (i.e., packed) and
      // so the "message type" is what is packed in each ciphertext, i.e., a
      // single dimensional tensor corresponding to the last axis.
      // TODO(#2280): this is where we are forced to use the packed cleartexts
      messageType = RankedTensorType::get(
          {tensorValueType.getDimSize(tensorValueType.getRank() - 1)},
          tensorValueType.getElementType());
    }

    auto ctType = lwe::LWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, messageType,
                                      lwe::NoOverflowAttr::get(ctx)),
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing,
            lwe::InverseCanonicalEncodingAttr::get(ctx, scale)),
        lwe::CiphertextSpaceAttr::get(ctx,
                                      getRlweRNSRingWithLevel(ring_, level),
                                      lwe::LweEncryptionType::mix, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, modulusChain, level));

    if (tensorValueType) {
      // A rank-1 tensor is interpreted as a single ciphertext
      if (tensorValueType.getRank() == 1) {
        return ctType;
      }

      // A rank-2+ tensor is a tensor of ciphertexts, where the last axis
      // becomes the ciphertext type. This is the most common case where
      // data is packed in a set of ciphertexts.
      return RankedTensorType::get(tensorValueType.getShape().drop_back(),
                                   ctType);
    }

    return ctType;
  }

 private:
  polynomial::RingAttr ring_;
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

    // Invariant: for every SecretType, there is a
    // corresponding MgmtAttr attached to it,
    // either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp).
    // Otherwise the typeConverter won't find the proper type information
    // and fail
    SecretToCKKSTypeConverter typeConverter(context, rlweRing.value());
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<ckks::CKKSDialect, lwe::LWEDialect,
                           arith::ArithDialect, tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<mgmt::ModReduceOp, mgmt::RelinearizeOp>();

    target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineYieldOp>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::CallOp>(
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
        ConvertExtractSlice, ConvertInsertSlice,
        ConvertAnyContextAware<affine::AffineForOp>,
        ConvertAnyContextAware<affine::AffineYieldOp>,
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
    mgmt::cleanupInitOp(getOperation());
  }
};

}  // namespace mlir::heir
