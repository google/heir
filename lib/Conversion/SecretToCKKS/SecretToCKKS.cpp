#include "lib/Conversion/SecretToCKKS/SecretToCKKS.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
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

#define GEN_PASS_DEF_SECRETTOCKKS
#include "lib/Conversion/SecretToCKKS/SecretToCKKS.h.inc"

namespace {

// Returns an RLWE ring given the specified number of bits needed and polynomial
// modulus degree.
// TODO(#536): Integrate a general library to compute appropriate prime moduli
// given any number of bits.
FailureOr<::mlir::polynomial::RingAttr> getRlweRing(MLIRContext *ctx,
                                                    int coefficientModBits,
                                                    int polyModDegree) {
  std::vector<::mlir::polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, polyModDegree);
  monomials.emplace_back(1, 0);
  auto result = ::mlir::polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  ::mlir::polynomial::IntPolynomial xnPlusOne = result.value();
  switch (coefficientModBits) {
    case 29: {
      auto type = IntegerType::get(ctx, 32);
      return ::mlir::polynomial::RingAttr::get(
          type, IntegerAttr::get(type, APInt(32, 463187969)),
          polynomial::IntPolynomialAttr::get(ctx, xnPlusOne));
    }
    default:
      return failure();
  }
}

}  // namespace

// Remove this class if no type conversions are necessary
class SecretToCKKSTypeConverter : public TypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext *ctx,
                            ::mlir::polynomial::RingAttr rlweRing) {
    addConversion([](Type type) { return type; });

    // Convert secret types to LWE ciphertext types.
    addConversion([ctx, this](secret::SecretType type) -> Type {
      // TODO(#785): Set a scaling parameter for floating point values.
      int bitWidth =
          llvm::TypeSwitch<Type, int>(type.getValueType())
              .Case<RankedTensorType>(
                  [&](auto ty) -> int { return ty.getElementTypeBitWidth(); })
              .Case<IntegerType>([&](auto ty) -> int { return ty.getWidth(); });
      return lwe::RLWECiphertextType::get(
          ctx,
          lwe::PolynomialEvaluationEncodingAttr::get(ctx, bitWidth, bitWidth),
          lwe::RLWEParamsAttr::get(ctx, 2, ring_), type.getValueType());
    });

    ring_ = rlweRing;
  }

 private:
  ::mlir::polynomial::RingAttr ring_;
};

struct SecretToCKKS : public impl::SecretToCKKSBase<SecretToCKKS> {
  using SecretToCKKSBase::SecretToCKKSBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    auto rlweRing = getRlweRing(context, coefficientModBits, polyModDegree);
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

    SecretToCKKSTypeConverter typeConverter(context, rlweRing.value());
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<ckks::CKKSDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<secret::GenericOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);
    patterns.add<
        SecretGenericOpCipherConversion<arith::AddIOp, ckks::AddOp>,
        SecretGenericOpCipherConversion<arith::SubIOp, ckks::SubOp>,
        SecretGenericOpCipherConversion<arith::AddFOp, ckks::AddOp>,
        SecretGenericOpCipherConversion<arith::SubFOp, ckks::SubOp>,
        SecretGenericOpConversion<tensor::ExtractOp, ckks::ExtractOp>,
        SecretGenericOpRotateConversion<ckks::RotateOp>,
        SecretGenericOpMulConversion<arith::MulIOp, ckks::MulOp,
                                     ckks::RelinearizeOp>,
        SecretGenericOpMulConversion<arith::MulFOp, ckks::MulOp,
                                     ckks::RelinearizeOp>,
        SecretGenericOpCipherPlainConversion<arith::AddFOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubFOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulFOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, ckks::MulPlainOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
