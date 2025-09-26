#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
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
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "secret-to-bgv"

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOBGV
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h.inc"

auto& kArgMgmtAttrName = mgmt::MgmtDialect::kArgMgmtAttrName;

namespace {

// Returns an RLWE RNS ring given the specified number of bits needed and
// polynomial modulus degree.
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
  for (int64_t prime : primes) {
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

class SecretToBGVTypeConverter : public TypeConverter {
 public:
  SecretToBGVTypeConverter(MLIRContext* ctx, polynomial::RingAttr rlweRing,
                           int64_t ptm, bool isBFV)
      : ring(rlweRing), plaintextModulus(ptm), isBFV(isBFV) {
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
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();
    auto scale = mgmtAttr.getScale();

    auto* ctx = type.getContext();
    auto plaintextRing = polynomial::RingAttr::get(
        type.getContext(),
        mod_arith::ModArithType::get(
            ctx, IntegerAttr::get(IntegerType::get(ctx, 64), plaintextModulus)),
        ring.getPolynomialModulus());

    SmallVector<IntegerAttr> moduliChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      moduliChain.push_back(modulus);
    }

    auto encryptionType =
        isBFV ? lwe::LweEncryptionType::msb : lwe::LweEncryptionType::lsb;

    return lwe::LWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, type.getValueType(),
                                      lwe::NoOverflowAttr::get(ctx)),
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing,
            lwe::FullCRTPackingEncodingAttr::get(ctx, scale)),
        lwe::CiphertextSpaceAttr::get(ctx, getRlweRNSRingWithLevel(ring, level),
                                      encryptionType, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, moduliChain, level));
  }

 private:
  polynomial::RingAttr ring;
  int64_t plaintextModulus;
  bool isBFV;
};

LogicalResult disallowFloatlike(const Type& type) {
  auto secretType = dyn_cast<secret::SecretType>(type);
  if (!secretType) return success();

  if (isa<FloatType>(getElementTypeOrSelf(secretType.getValueType())))
    return failure();

  return success();
}

struct SecretToBGV : public impl::SecretToBGVBase<SecretToBGV> {
  using SecretToBGVBase::SecretToBGVBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    auto schemeParamAttr = module->getAttrOfType<bgv::SchemeParamAttr>(
        bgv::BGVDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      module->emitError("expected BGV scheme parameters");
      signalPassFailure();
      return;
    }

    bool usePublicKey =
        schemeParamAttr.getEncryptionType() == bgv::BGVEncryptionType::pk;

    // NOTE: 2 ** logN != polyModDegree
    // they have different semantic
    // auto logN = schemeParamAttr.getLogN();
    auto plaintextModulus = schemeParamAttr.getPlaintextModulus();

    // pass option polyModDegree is actually the number of slots
    // TODO(#1402): use a proper name for BGV
    auto rlweRing = getRlweRNSRing(context, schemeParamAttr.getQ().asArrayRef(),
                                   polyModDegree);
    if (failed(rlweRing)) {
      return signalPassFailure();
    }
    // Ensure that all secret types are uniform and matching the ring
    // parameter size.
    Operation* foundOp = walkAndDetect(module, [&](Operation* op) {
      ValueRange valuesToCheck = op->getOperands();
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        valuesToCheck = funcOp.getArguments();
      }
      for (auto value : valuesToCheck) {
        if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
          auto tensorTy = dyn_cast<RankedTensorType>(secretTy.getValueType());
          if (tensorTy && tensorTy.getShape() !=
                              ArrayRef<int64_t>{rlweRing.value()
                                                    .getPolynomialModulus()
                                                    .getPolynomial()
                                                    .getDegree()}) {
            return true;
          }
        }
      }
      return false;
    });
    if (foundOp != nullptr) {
      foundOp->emitError(
          "expected batched secret types to be tensors with dimension "
          "matching ring parameter");
      signalPassFailure();
      return;
    }

    if (failed(walkAndValidateTypes<secret::GenericOp>(
            module, disallowFloatlike,
            "Floating point types are not supported in BGV. Maybe you meant "
            "to use a CKKS pipeline like --mlir-to-ckks?"))) {
      signalPassFailure();
      return;
    }

    // Invariant: for every SecretType, there is a corresponding MgmtAttr
    // attached to it, either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp). Otherwise the typeConverter won't find the proper
    // type information and fail
    SecretToBGVTypeConverter typeConverter(context, rlweRing.value(),
                                           plaintextModulus,
                                           moduleIsBFV(getOperation()));

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<bgv::BGVDialect>();
    target.addLegalDialect<lwe::LWEDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<mgmt::ModReduceOp, mgmt::RelinearizeOp>();
    target.addIllegalOp<secret::GenericOp>();

    patterns.add<
        SecretGenericOpConversion<arith::AddIOp, bgv::AddOp>,
        SecretGenericOpConversion<arith::SubIOp, bgv::SubOp>,
        SecretGenericOpConversion<arith::MulIOp, bgv::MulOp>,
        SecretGenericOpConversion<arith::ExtUIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpConversion<arith::ExtSIOp,
                                  lwe::ReinterpretApplicationDataOp>,
        SecretGenericOpRelinearizeConversion<bgv::RelinearizeOp>,
        SecretGenericOpModulusSwitchConversion<bgv::ModulusSwitchOp>,
        SecretGenericOpRotateConversion<bgv::RotateColumnsOp>,
        SecretGenericOpLevelReduceConversion<bgv::LevelReduceOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, bgv::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, bgv::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, bgv::MulPlainOp>,
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
