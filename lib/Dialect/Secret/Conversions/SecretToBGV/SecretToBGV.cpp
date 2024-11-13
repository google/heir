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
#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
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
class SecretToBGVTypeConverter : public TypeConverter {
 public:
  SecretToBGVTypeConverter(MLIRContext *ctx,
                           ::mlir::polynomial::RingAttr rlweRing) {
    addConversion([](Type type) { return type; });

    // Convert secret types to BGV ciphertext types
    addConversion([ctx, this](secret::SecretType type) -> Type {
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

struct SecretToBGV : public impl::SecretToBGVBase<SecretToBGV> {
  using SecretToBGVBase::SecretToBGVBase;

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

    SecretToBGVTypeConverter typeConverter(context, rlweRing.value());
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<bgv::BGVDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<secret::GenericOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);
    patterns.add<
        SecretGenericOpCipherConversion<arith::AddIOp, bgv::AddOp>,
        SecretGenericOpCipherConversion<arith::SubIOp, bgv::SubOp>,
        SecretGenericOpConversion<tensor::ExtractOp, bgv::ExtractOp>,
        SecretGenericOpRotateConversion<bgv::RotateOp>,
        SecretGenericOpMulConversion<arith::MulIOp, bgv::MulOp,
                                     bgv::RelinearizeOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, bgv::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, bgv::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, bgv::MulPlainOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
