#include "lib/Dialect/LWE/Conversions/LWEToStableHLO/LWEToStableHLO.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"                  // from @stablehlo

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOSTABLEHLO
#include "lib/Dialect/LWE/Conversions/LWEToStableHLO/LWEToStableHLO.h.inc"

namespace {

bool isLWEElementType(Type type) {
  Type eltTy = getElementTypeOrSelf(type);
  return isa<lwe::LWEDialect>(eltTy.getDialect());
}

// Returns the moduli of the ciphertext type.
// If the ciphertext type is not in RNS or ModArith, returns failure.
// If the ciphertext type is in RNS, returns the moduli of the basis types.
// If the ciphertext type is in ModArith, returns the modulus.
// If diagOp is provided, emits an error if the ciphertext type is not in RNS
// or ModArith, or if the ciphertext type has a modulus that is not in (1,
// 2^32).
FailureOr<SmallVector<uint64_t>> extractModuli(lwe::LWECiphertextType ctType,
                                               Operation* diagOp = nullptr) {
  Type coeffTy = ctType.getCiphertextSpace().getRing().getCoefficientType();
  SmallVector<mod_arith::ModArithType> limbTypes;
  // Limbs are either of type RNSType or ModArithType.
  if (auto rnsTy = dyn_cast<rns::RNSType>(coeffTy)) {
    for (Type basisTy : rnsTy.getBasisTypes()) {
      auto modTy = dyn_cast<mod_arith::ModArithType>(basisTy);
      if (!modTy) {
        if (diagOp) {
          diagOp->emitError()
              << "expected mod_arith limb type in RNS basis, got " << basisTy;
        }
        return failure();
      }
      limbTypes.push_back(modTy);
    }
  } else if (auto modTy = dyn_cast<mod_arith::ModArithType>(coeffTy)) {
    limbTypes.push_back(modTy);
  } else {
    if (diagOp) {
      diagOp->emitError() << "expected mod_arith or RNS coefficient type, got "
                          << coeffTy;
    }
    return failure();
  }

  if (limbTypes.empty()) {
    if (diagOp) {
      diagOp->emitError() << "ciphertext RNS basis must not be empty";
    }
    return failure();
  }

  SmallVector<uint64_t> moduli;
  moduli.reserve(limbTypes.size());
  for (mod_arith::ModArithType modTy : limbTypes) {
    const APInt& modVal = modTy.getModulus().getValue();
    if (modVal.ule(1) || modVal.getActiveBits() > 32) {
      if (diagOp) {
        diagOp->emitError()
            << "ui32 StableHLO lowering requires moduli in (1, 2^32), got "
            << modTy;
      }
      return failure();
    }
    moduli.push_back(modVal.getZExtValue());
  }
  return moduli;
}

// Factors the ring degree into (r, c) such that r * c = degree.
// If rOpt or cOpt are non-zero, they are used as hints for the factorization.
// Returns failure if:
// - degree is not a power of two.
// - rOpt or cOpt are not powers of two or zero.
// - rOpt * cOpt != degree.
FailureOr<std::pair<int64_t, int64_t>> factorRingDegree(
    uint32_t degree, int rOpt, int cOpt, Operation* diagOp = nullptr) {
  if (degree == 0 || !llvm::isPowerOf2_32(degree)) {
    if (diagOp) {
      diagOp->emitError()
          << "ciphertext polynomial degree must be a power of two, got "
          << degree;
    }
    return failure();
  }

  if (rOpt < 0 || cOpt < 0 || (rOpt > 0 && !llvm::isPowerOf2_64(rOpt)) ||
      (cOpt > 0 && !llvm::isPowerOf2_64(cOpt))) {
    if (diagOp) {
      diagOp->emitError()
          << "r and c must be non-negative powers of two (or 0 for auto), got "
             "r="
          << rOpt << ", c=" << cOpt;
    }
    return failure();
  }

  // For auto, try to make (r, c) close to a square.
  int64_t deg64 = static_cast<int64_t>(degree);
  int64_t rDim = rOpt;
  int64_t cDim = cOpt;
  if (rOpt == 0 && cOpt == 0) {
    rDim = 1LL << (llvm::Log2_32(degree) / 2);
    cDim = deg64 / rDim;
  } else if (rOpt > 0 && cOpt == 0) {
    cDim = deg64 / rDim;
  } else if (rOpt == 0 && cOpt > 0) {
    rDim = deg64 / cDim;
  }

  if (rDim <= 0 || cDim <= 0 || rDim * cDim != deg64) {
    if (diagOp) {
      diagOp->emitError() << "degree layout (r=" << rOpt << ", c=" << cOpt
                          << ") does not multiply to ring degree " << degree;
    }
    return failure();
  }
  return std::make_pair(rDim, cDim);
}

// Converts a ciphertext type to
// tensor<batchSize x numElements x r x c x numModuli>.
FailureOr<RankedTensorType> convertCiphertextToCanonicalRank5Tensor(
    lwe::LWECiphertextType ctType, int64_t batchSize, int rOpt, int cOpt,
    Operation* diagOp = nullptr) {
  if (batchSize <= 0) {
    if (diagOp) {
      diagOp->emitError() << "batch size must be positive, got " << batchSize;
    }
    return failure();
  }

  FailureOr<SmallVector<uint64_t>> moduli = extractModuli(ctType, diagOp);
  if (failed(moduli)) return failure();

  polynomial::IntPolynomialAttr polyMod =
      ctType.getCiphertextSpace().getRing().getPolynomialModulus();
  if (!polyMod) {
    if (diagOp) {
      diagOp->emitError()
          << "ciphertext ring must specify a polynomial modulus";
    }
    return failure();
  }

  uint32_t degree = polyMod.getPolynomial().getDegree();
  FailureOr<std::pair<int64_t, int64_t>> layout =
      factorRingDegree(degree, rOpt, cOpt, diagOp);
  if (failed(layout)) return failure();

  int64_t numElements = ctType.getCiphertextSpace().getSize();
  int64_t numModuli = static_cast<int64_t>(moduli->size());
  auto ui32Ty =
      IntegerType::get(ctType.getContext(), 32, IntegerType::Unsigned);
  return RankedTensorType::get(
      {batchSize, numElements, layout->first, layout->second, numModuli},
      ui32Ty);
}

// Converts an LWE type to the canonical rank-5 tensor type. Returns the
// original type if it is not an LWE type. The supported LWE types are:
// - LWECiphertextType
// - ShapedType<LWECiphertextType>
FailureOr<Type> convertToCanonicalRank5TensorOrSelf(
    Type type, int defaultBatch, int rOpt, int cOpt,
    Operation* diagOp = nullptr) {
  if (!isLWEElementType(type)) {
    return type;
  }
  // Single ciphertext.
  if (auto ctType = dyn_cast<lwe::LWECiphertextType>(type)) {
    FailureOr<RankedTensorType> res = convertCiphertextToCanonicalRank5Tensor(
        ctType, defaultBatch, rOpt, cOpt, diagOp);
    if (failed(res)) return failure();
    return Type(*res);
  }
  // Tensor of ciphertexts.
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    auto tensorTy = dyn_cast<RankedTensorType>(shapedTy);
    auto ctEltTy = dyn_cast<lwe::LWECiphertextType>(shapedTy.getElementType());
    if (!tensorTy || !ctEltTy || tensorTy.getRank() != 1 ||
        !tensorTy.hasStaticShape() || tensorTy.getDimSize(0) <= 0) {
      if (diagOp) {
        diagOp->emitError()
            << "expected 1D statically shaped tensor of LWE ciphertexts, got "
            << type;
      }
      return failure();
    }
    FailureOr<RankedTensorType> res = convertCiphertextToCanonicalRank5Tensor(
        ctEltTy, tensorTy.getDimSize(0), rOpt, cOpt, diagOp);
    if (failed(res)) return failure();
    return Type(*res);
  }
  if (diagOp) {
    diagOp->emitError() << "unsupported LWE type in StableHLO lowering: "
                        << type;
  }
  return failure();
}

class CanonicalRank5TensorTypeConverter : public TypeConverter {
 public:
  CanonicalRank5TensorTypeConverter(int defaultBatch, int rOpt, int cOpt) {
    addConversion([defaultBatch, rOpt, cOpt](Type type) -> Type {
      FailureOr<Type> converted =
          convertToCanonicalRank5TensorOrSelf(type, defaultBatch, rOpt, cOpt);
      return succeeded(converted) ? *converted : Type();
    });
  }
};

struct ConvertRAddOp : public OpConversionPattern<lwe::RAddOp> {
  using OpConversionPattern<lwe::RAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getLhs().getType()));
    if (!ctType) {
      return rewriter.notifyMatchFailure(op, "expected LWE ciphertext operand");
    }

    FailureOr<SmallVector<uint64_t>> moduli = extractModuli(ctType);
    if (failed(moduli)) {
      return rewriter.notifyMatchFailure(op, "unsupported ciphertext moduli");
    }

    Type expectedResultTy =
        getTypeConverter()->convertType(op.getResult().getType());
    Value lhs32 = adaptor.getLhs();
    Value rhs32 = adaptor.getRhs();
    auto u32TensorTy = dyn_cast<RankedTensorType>(lhs32.getType());
    if (!u32TensorTy || u32TensorTy != rhs32.getType() ||
        u32TensorTy != expectedResultTy || u32TensorTy.getRank() != 5) {
      return rewriter.notifyMatchFailure(
          op, "expected matching rank-5 ui32 ciphertext tensors");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext* ctx = b.getContext();
    auto ui64Ty = IntegerType::get(ctx, 64, IntegerType::Unsigned);
    auto u64TensorTy = RankedTensorType::get(u32TensorTy.getShape(), ui64Ty);

    Value lhs64 = stablehlo::ConvertOp::create(b, u64TensorTy, lhs32);
    Value rhs64 = stablehlo::ConvertOp::create(b, u64TensorTy, rhs32);
    Value sum64 = stablehlo::AddOp::create(b, lhs64, rhs64);

    auto q1DTy =
        RankedTensorType::get({static_cast<int64_t>(moduli->size())}, ui64Ty);
    SmallVector<APInt> qVals;
    qVals.reserve(moduli->size());
    for (uint64_t q : *moduli) {
      qVals.emplace_back(/*numBits=*/64, q, /*isSigned=*/false);
    }
    Value q1D =
        stablehlo::ConstantOp::create(b, DenseElementsAttr::get(q1DTy, qVals));

    // StableHLO does not implicitly broadcast, so broadcast primes tensor.
    // Align primes with dim 4 (num_moduli).
    Value q5D = stablehlo::BroadcastInDimOp::create(
        b, u64TensorTy, q1D, b.getDenseI64ArrayAttr({4}));

    // %reduced64 = (%sum64 >= %q5D) ? (%sum64 - %q5D) : %sum64
    Value geMask = stablehlo::CompareOp::create(
        b, sum64, q5D, stablehlo::ComparisonDirection::GE,
        stablehlo::ComparisonType::UNSIGNED);
    Value diff64 = stablehlo::SubtractOp::create(b, sum64, q5D);
    Value reduced64 = stablehlo::SelectOp::create(b, geMask, diff64, sum64);
    Value res32 = stablehlo::ConvertOp::create(b, u32TensorTy, reduced64);

    rewriter.replaceOp(op, res32);
    return success();
  }
};

// Validates that all LWE types under `root` can lower to canonical rank 5
// StableHLO tensors and emits errors on the first failure.
LogicalResult validateCanonicalRank5TensorTypes(Operation* root,
                                                int defaultBatch, int rOpt,
                                                int cOpt) {
  auto checkType = [&](Type ty, Operation* diagOp) -> LogicalResult {
    return failure(failed(convertToCanonicalRank5TensorOrSelf(
        ty, defaultBatch, rOpt, cOpt, diagOp)));
  };

  WalkResult validation = root->walk([&](Operation* op) -> WalkResult {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      for (Type inTy : funcOp.getFunctionType().getInputs()) {
        if (failed(checkType(inTy, funcOp))) return WalkResult::interrupt();
      }
      for (Type resTy : funcOp.getFunctionType().getResults()) {
        if (failed(checkType(resTy, funcOp))) return WalkResult::interrupt();
      }
    }
    for (Type operandTy : op->getOperandTypes()) {
      if (failed(checkType(operandTy, op))) return WalkResult::interrupt();
    }
    for (Type resTy : op->getResultTypes()) {
      if (failed(checkType(resTy, op))) return WalkResult::interrupt();
    }
    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (failed(checkType(arg.getType(), op)))
            return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(validation.wasInterrupted());
}

struct LWEToStableHLO : public impl::LWEToStableHLOBase<LWEToStableHLO> {
  using LWEToStableHLOBase::LWEToStableHLOBase;

  void runOnOperation() override {
    Operation* root = getOperation();
    if (batch <= 0) {
      root->emitError() << "batch size must be positive, got " << batch;
      return signalPassFailure();
    }
    if (r < 0 || c < 0 || (r > 0 && !llvm::isPowerOf2_64(r)) ||
        (c > 0 && !llvm::isPowerOf2_64(c))) {
      root->emitError()
          << "r and c must be non-negative powers of two (or 0 for auto), got "
             "r="
          << r << ", c=" << c;
      return signalPassFailure();
    }

    if (failed(validateCanonicalRank5TensorTypes(root, batch, r, c))) {
      return signalPassFailure();
    }

    MLIRContext* ctx = &getContext();
    CanonicalRank5TensorTypeConverter typeConverter(batch, r, c);

    ConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertRAddOp>(typeConverter, ctx);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      return typeConverter.isLegal(op) &&
             (isNotBranchOpInterfaceOrReturnLikeOp(op) ||
              isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                               typeConverter) ||
              isLegalForReturnOpTypeConversionPattern(op, typeConverter));
    });

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyFullConversion(root, target, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::heir::lwe
