#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
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
    int bitWidth =
        llvm::TypeSwitch<Type, int>(type.getValueType())
            .Case<RankedTensorType>(
                [&](auto ty) -> int { return ty.getElementTypeBitWidth(); })
            .Case<IntegerType>([&](auto ty) -> int { return ty.getWidth(); });

    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();

    auto *ctx = type.getContext();
    return lwe::RLWECiphertextType::get(
        ctx,
        lwe::PolynomialEvaluationEncodingAttr::get(ctx, bitWidth, bitWidth),
        lwe::RLWEParamsAttr::get(ctx, dimension,
                                 getRlweRNSRingWithLevel(ring, level)),
        type.getValueType());
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

struct SecretToBGV : public impl::SecretToBGVBase<SecretToBGV> {
  using SecretToBGVBase::SecretToBGVBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    auto maxLevel = 5;
    auto rlweRing =
        getRlweRNSRing(context, maxLevel, coefficientModBits, polyModDegree);
    if (failed(rlweRing)) {
      return signalPassFailure();
    }
    // Ensure that all secret types are uniform and matching the ring
    // parameter size.
    WalkResult compatibleTensors = module->walk([&](Operation *op) {
      for (auto value : op->getOperands()) {
        if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
          auto tensorTy = dyn_cast<RankedTensorType>(secretTy.getValueType());
          if (tensorTy && tensorTy.getShape() !=
                              ArrayRef<int64_t>{rlweRing.value()
                                                    .getPolynomialModulus()
                                                    .getPolynomial()
                                                    .getDegree()}) {
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (compatibleTensors.wasInterrupted()) {
      module->emitError(
          "expected batched secret types to be tensors with dimension "
          "matching ring parameter");
      return signalPassFailure();
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
