#include "lib/Conversion/SecretToBGV/SecretToBGV.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
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

#define GEN_PASS_DEF_SECRETTOBGV
#include "lib/Conversion/SecretToBGV/SecretToBGV.h.inc"

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

template <typename T, typename Y>
class SecretGenericOpConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // arithmetic ops.
      return failure();
    }

    auto &innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // Assemble the arguments for the BGV operation.
    SmallVector<Value> inputs;
    for (OpOperand &operand : innerOp.getOpOperands()) {
      if (auto *secretArg = op.getOpOperandForBlockArgument(operand.get())) {
        inputs.push_back(
            adaptor.getODSOperands(0)[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand.get());
      }
    }

    // Directly convert the op if all operands are ciphertext.
    SmallVector<Type> resultTypes;
    auto result =
        getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes);
    if (failed(result)) return failure();

    replaceOp(op, resultTypes, inputs, rewriter);
    return success();
  }

  // Default method for replacing the secret.generic with the target operation.
  virtual void replaceOp(secret::GenericOp op, TypeRange outputTypes,
                         ValueRange inputs,
                         ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs);
  }
};

class SecretGenericOpMulConversion
    : public SecretGenericOpConversion<arith::MulIOp, bgv::MulOp> {
 public:
  using SecretGenericOpConversion<arith::MulIOp,
                                  bgv::MulOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bgv::Relinearize>(
        op, rewriter.create<bgv::MulOp>(op.getLoc(), inputs),
        rewriter.getDenseI32ArrayAttr({0, 1, 2}),
        rewriter.getDenseI32ArrayAttr({0, 1}));
  }
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
    patterns.add<SecretGenericOpConversion<arith::AddIOp, bgv::AddOp>,
                 SecretGenericOpConversion<arith::SubIOp, bgv::SubOp>,
                 SecretGenericOpConversion<tensor::ExtractOp, bgv::ExtractOp>,
                 SecretGenericOpConversion<tensor_ext::RotateOp, bgv::RotateOp>,
                 SecretGenericOpMulConversion>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
