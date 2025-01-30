#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOBGV
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h.inc"

namespace {

// Returns an RLWE RNS ring given the specified number of bits needed and
// polynomial modulus degree.
// TODO(#536): Integrate a general library to compute appropriate prime moduli
// given any number of bits.
FailureOr<polynomial::RingAttr> getRlweRNSRing(MLIRContext *ctx,
                                               int currentLevel,
                                               int coefficientModBits,
                                               int polyModDegree) {
  std::vector<::mlir::heir::polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, polyModDegree);
  monomials.emplace_back(1, 0);
  auto result =
      ::mlir::heir::polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  ::mlir::heir::polynomial::IntPolynomial xnPlusOne = result.value();
  // all 40 bit primes...
  std::vector<int64_t> primes = {1095233372161, 1032955396097, 1005037682689,
                                 998595133441,  972824936449,  959939837953};
  SmallVector<Type, 4> modTypes;
  for (int i = 0; i <= currentLevel; i++) {
    auto type = IntegerType::get(ctx, 64);
    modTypes.push_back(
        mod_arith::ModArithType::get(ctx, IntegerAttr::get(type, primes[i])));
  }
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

}  // namespace

class SecretToBGVTypeConverter : public TypeWithAttrTypeConverter {
 public:
  SecretToBGVTypeConverter(MLIRContext *ctx,
                           ::mlir::heir::polynomial::RingAttr rlweRing)
      : TypeWithAttrTypeConverter(mgmt::MgmtDialect::kArgMgmtAttrName) {
    ring = rlweRing;

    // isLegal/isSignatureLegal will always be true
    addConversion([](Type type) { return type; });
  }

  Type convertSecretTypeWithMgmtAttr(secret::SecretType type,
                                     mgmt::MgmtAttr mgmtAttr) const {
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();

    auto *ctx = type.getContext();
    // TODO(#661) : Calculate the appropriate values by analyzing the function
    auto plaintextRing = ::mlir::heir::polynomial::RingAttr::get(
        type.getContext(),
        mod_arith::ModArithType::get(
            ctx, IntegerAttr::get(IntegerType::get(ctx, 64), 4295294977)),
        ring.getPolynomialModulus());

    SmallVector<IntegerAttr, 6> moduliChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      moduliChain.push_back(modulus);
    }

    return lwe::NewLWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, type.getValueType(),
                                      lwe::NoOverflowAttr::get(ctx)),
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing, lwe::FullCRTPackingEncodingAttr::get(ctx, 0)),
        lwe::CiphertextSpaceAttr::get(ctx, getRlweRNSRingWithLevel(ring, level),
                                      lwe::LweEncryptionType::lsb, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, moduliChain, level));
  }

  Type convertTypeWithAttr(Type type, Attribute attr) const override {
    auto secretTy = dyn_cast<secret::SecretType>(type);
    // guard against null attribute
    if (secretTy && attr) {
      auto mgmtAttr = dyn_cast<mgmt::MgmtAttr>(attr);
      if (mgmtAttr) {
        return convertSecretTypeWithMgmtAttr(secretTy, mgmtAttr);
      }
    }
    return type;
  }

 private:
  ::mlir::heir::polynomial::RingAttr ring;
};

LogicalResult disallowFloatlike(const Type &type) {
  auto secretType = dyn_cast<secret::SecretType>(type);
  if (!secretType) return success();

  if (isa<FloatType>(getElementTypeOrSelf(secretType.getValueType())))
    return failure();

  return success();
}

struct SecretToBGV : public impl::SecretToBGVBase<SecretToBGV> {
  using SecretToBGVBase::SecretToBGVBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // Helper for future lowerings that want to know what scheme was used
    module->setAttr(kBGVSchemeAttrName, UnitAttr::get(context));

    auto maxLevel = 5;
    auto rlweRing =
        getRlweRNSRing(context, maxLevel, coefficientModBits, polyModDegree);
    if (failed(rlweRing)) {
      return signalPassFailure();
    }
    // Ensure that all secret types are uniform and matching the ring
    // parameter size.
    Operation *foundOp = walkAndDetect(module, [&](Operation *op) {
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

    // Invariant: for every SecretType, there is a
    // corresponding MgmtAttr attached to it,
    // either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp).
    // Otherwise the typeConverter won't find the proper type information
    // and fail
    SecretToBGVTypeConverter typeConverter(context, rlweRing.value());

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<bgv::BGVDialect>();
    target.addLegalDialect<lwe::LWEDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<mgmt::ModReduceOp, mgmt::RelinearizeOp>();
    target.addIllegalOp<secret::GenericOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isFuncArgumentAndResultLegal(op);
    });

    patterns.add<
        ConvertFuncWithContextAwareTypeConverter,
        SecretGenericOpCipherConversion<arith::AddIOp, bgv::AddOp>,
        SecretGenericOpCipherConversion<arith::SubIOp, bgv::SubOp>,
        SecretGenericOpCipherConversion<arith::MulIOp, bgv::MulOp>,
        SecretGenericOpRelinearizeConversion<bgv::RelinearizeOp>,
        SecretGenericOpModulusSwitchConversion<bgv::ModulusSwitchOp>,
        SecretGenericOpConversion<tensor::ExtractOp, bgv::ExtractOp>,
        SecretGenericOpRotateConversion<bgv::RotateOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, bgv::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, bgv::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, bgv::MulPlainOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // cleanup MgmtAttr
    getOperation()->walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
          funcOp.removeArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName);
        }
      } else {
        op->removeAttr(mgmt::MgmtDialect::kArgMgmtAttrName);
      }
    });
  }
};

}  // namespace mlir::heir
