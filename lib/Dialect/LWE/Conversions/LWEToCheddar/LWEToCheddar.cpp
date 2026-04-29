#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "lwe-to-cheddar"

namespace mlir::heir::lwe {

static FailureOr<int64_t> getCKKSLogDefaultScale(Operation* op) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return failure();
  }
  auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    return failure();
  }
  return schemeParamAttr.getLogDefaultScale();
}

enum class CheddarLevelReduceBucketKind {
  kCanonical,
  kScaledOnce,
};

static FailureOr<CheddarLevelReduceBucketKind> getCheddarLevelReduceBucketKind(
    Operation* op, lwe::LWECiphertextType ctType) {
  auto logDefaultScale = getCKKSLogDefaultScale(op);
  if (failed(logDefaultScale)) {
    return failure();
  }

  // Scaling factors on CKKS encodings are stored log-additively on main (a
  // multiply adds the log-scales). The "canonical" scale after the initial
  // encoding is `logDefaultScale`; one post-multiply-without-rescale bucket
  // is `2 * logDefaultScale`.
  int64_t logScale = getScalingFactorFromEncodingAttr(
      ctType.getPlaintextSpace().getEncoding());

  if (logScale == *logDefaultScale) {
    return CheddarLevelReduceBucketKind::kCanonical;
  }
  if (logScale == 2 * *logDefaultScale) {
    return CheddarLevelReduceBucketKind::kScaledOnce;
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

class ToCheddarTypeConverter : public TypeConverter {
 public:
  ToCheddarTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      return cheddar::CiphertextType::get(ctx);
    });
    addConversion([ctx](lwe::LWEPlaintextType type) -> Type {
      return cheddar::PlaintextType::get(ctx);
    });
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      LLVM_DEBUG(llvm::dbgs()
                 << "Converting LWEPublicKeyType -> UserInterfaceType\n");
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      LLVM_DEBUG(llvm::dbgs()
                 << "Converting LWESecretKeyType -> UserInterfaceType\n");
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([this](RankedTensorType type) -> Type {
      return RankedTensorType::get(type.getShape(),
                                   this->convertType(type.getElementType()));
    });
  }
};

//===----------------------------------------------------------------------===//
// Helper: get contextual arguments by type
//===----------------------------------------------------------------------===//

namespace {

template <typename... Dialects>
bool containsArgumentOfDialect(Operation* op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    return false;
  }
  return llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
    return DialectEqual<Dialects...>()(
        &getElementTypeOrSelf(argType).getDialect());
  });
}

template <typename CheddarType>
FailureOr<Value> getContextualArg(Operation* op) {
  auto result = getContextualArgFromFunc<CheddarType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found op in a function without a required CHEDDAR context "
              "argument. Did the AddEvaluatorArg pattern fail to run?";
  }
  return result.value();
}

FailureOr<Value> getContextualArg(Operation* op, Type type) {
  return getContextualArgFromFunc(op, type);
}

SmallVector<Type> getRequiredCheddarContextTypes(
    Operation* op,
    const std::vector<std::pair<Type, OpPredicate>>& evaluators) {
  SmallVector<Type> requiredTypes;
  for (const auto& evaluator : evaluators) {
    if (evaluator.second(op)) {
      requiredTypes.push_back(evaluator.first);
    }
  }
  return requiredTypes;
}

bool hasLeadingTypes(TypeRange actualTypes, ArrayRef<Type> requiredTypes) {
  if (actualTypes.size() < requiredTypes.size()) {
    return false;
  }
  return llvm::equal(requiredTypes,
                     actualTypes.take_front(requiredTypes.size()));
}

void addRequiredCheddarContextArgs(
    ModuleOp module,
    const std::vector<std::pair<Type, OpPredicate>>& evaluators) {
  module.walk([&](func::FuncOp funcOp) {
    SmallVector<Type> requiredTypes =
        getRequiredCheddarContextTypes(funcOp, evaluators);
    if (requiredTypes.empty() ||
        hasLeadingTypes(funcOp.getArgumentTypes(), requiredTypes)) {
      return;
    }

    SmallVector<unsigned> argIndices(requiredTypes.size(), 0);
    SmallVector<DictionaryAttr> argAttrs(requiredTypes.size(), nullptr);
    SmallVector<Location> argLocs(requiredTypes.size(), funcOp.getLoc());
    (void)funcOp.insertArguments(argIndices, requiredTypes, argAttrs, argLocs);
  });
}

LogicalResult addRequiredCheddarContextOperandsToCalls(
    ModuleOp module,
    const std::vector<std::pair<Type, OpPredicate>>& evaluators) {
  LogicalResult result = success();
  module.walk([&](func::CallOp callOp) {
    if (failed(result)) {
      return WalkResult::interrupt();
    }

    auto callee = getCalledFunction(callOp);
    if (failed(callee)) {
      result = callOp.emitOpError("could not find callee function");
      return WalkResult::interrupt();
    }

    SmallVector<Type> requiredTypes =
        getRequiredCheddarContextTypes(callee.value(), evaluators);
    if (requiredTypes.empty() ||
        hasLeadingTypes(callOp.getOperandTypes(), requiredTypes)) {
      return WalkResult::advance();
    }

    SmallVector<Value> contextOperands;
    for (Type requiredType : requiredTypes) {
      auto contextOperand =
          getContextualArg(callOp.getOperation(), requiredType);
      if (failed(contextOperand)) {
        result = failure();
        return WalkResult::interrupt();
      }
      contextOperands.push_back(contextOperand.value());
    }

    callOp->insertOperands(0, contextOperands);
    return WalkResult::advance();
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

// Binary ct-ct operations: ckks.add -> cheddar.add, etc.
template <typename CKKSOp, typename CheddarOp>
struct ConvertCKKSBinOp : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;

    rewriter.replaceOpWithNewOp<CheddarOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// LWE R* ops (used by torch-linalg-to-ckks pipeline)
using ConvertRAddOp = ConvertCKKSBinOp<lwe::RAddOp, cheddar::AddOp>;
using ConvertRSubOp = ConvertCKKSBinOp<lwe::RSubOp, cheddar::SubOp>;
using ConvertRMulOp = ConvertCKKSBinOp<lwe::RMulOp, cheddar::MultOp>;

// Ct-pt operations
template <typename CKKSOp, typename CheddarOp>
struct ConvertCKKSPlainOp : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;

    // Ensure ciphertext is first operand (CHEDDAR convention)
    Value ciphertext = adaptor.getLhs();
    Value plaintext = adaptor.getRhs();
    if (!isa<cheddar::CiphertextType>(adaptor.getLhs().getType())) {
      ciphertext = adaptor.getRhs();
      plaintext = adaptor.getLhs();
    }

    rewriter.replaceOpWithNewOp<CheddarOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), ciphertext, plaintext);
    return success();
  }
};

using ConvertRAddPlainOp =
    ConvertCKKSPlainOp<lwe::RAddPlainOp, cheddar::AddPlainOp>;
using ConvertRMulPlainOp =
    ConvertCKKSPlainOp<lwe::RMulPlainOp, cheddar::MultPlainOp>;

// SubPlain needs special handling: when plaintext is LHS (pt - ct),
// we emit neg(ct) + pt instead of sub_plain with swapped operands.
template <typename SubPlainOp>
struct ConvertSubPlainOp : public OpConversionPattern<SubPlainOp> {
  using OpConversionPattern<SubPlainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubPlainOp op, typename SubPlainOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;

    auto outType = this->typeConverter->convertType(op.getOutput().getType());

    if (isa<cheddar::CiphertextType>(adaptor.getLhs().getType())) {
      // ct - pt: use sub_plain directly
      rewriter.replaceOpWithNewOp<cheddar::SubPlainOp>(
          op, outType, ctx.value(), adaptor.getLhs(), adaptor.getRhs());
    } else {
      // pt - ct: emit neg(ct) + add_plain(neg_ct, pt)
      auto negated = cheddar::NegOp::create(rewriter, op.getLoc(), outType,
                                            ctx.value(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<cheddar::AddPlainOp>(
          op, outType, ctx.value(), negated, adaptor.getLhs());
    }
    return success();
  }
};

using ConvertRSubPlainOp = ConvertSubPlainOp<lwe::RSubPlainOp>;

// Negate
struct ConvertCKKSNegateOp : public OpConversionPattern<ckks::NegateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::NegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;

    rewriter.replaceOpWithNewOp<cheddar::NegOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

// Relinearize — needs the multiplication key
struct ConvertCKKSRelinOp : public OpConversionPattern<ckks::RelinearizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    // Get the multiplication key from the UI
    auto multKey = cheddar::GetMultKeyOp::create(
        rewriter, op.getLoc(), cheddar::EvalKeyType::get(getContext()),
        ui.value());

    rewriter.replaceOpWithNewOp<cheddar::RelinearizeOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), multKey);
    return success();
  }
};

// Rescale (mod reduce in CKKS)
struct ConvertCKKSRescaleOp : public OpConversionPattern<ckks::RescaleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RescaleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;

    rewriter.replaceOpWithNewOp<cheddar::RescaleOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

// Rotate
struct ConvertCKKSRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift) {
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    }

    if (dynamicShift) {
      // Dynamic shift: the emitter will inline ui.GetRotationKey(shift).
      // Create a placeholder GetRotKeyOp that the emitter traces back to
      // the UserInterface for the key lookup.
      auto rotKey = cheddar::GetRotKeyOp::create(
          rewriter, op.getLoc(), cheddar::EvalKeyType::get(getContext()),
          ui.value(),
          rewriter.getI64IntegerAttr(
              cheddar::kDynamicRotationKeyDistanceSentinel));
      rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
          op, this->typeConverter->convertType(op.getOutput().getType()),
          ctx.value(), adaptor.getInput(), rotKey, dynamicShift,
          /*static_shift=*/nullptr);
    } else {
      // Static shift: get the rotation key at lowering time.
      auto rotKey = cheddar::GetRotKeyOp::create(
          rewriter, op.getLoc(), cheddar::EvalKeyType::get(getContext()),
          ui.value(), staticShift);
      rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
          op, this->typeConverter->convertType(op.getOutput().getType()),
          ctx.value(), adaptor.getInput(), rotKey, /*dynamic_shift=*/Value(),
          staticShift);
    }
    return success();
  }
};

// Level reduce
struct ConvertCKKSLevelReduceOp
    : public OpConversionPattern<ckks::LevelReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::LevelReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;

    // Derive target level from the output ciphertext's modulus chain.
    auto outputCtType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getOutput().getType()));
    auto inputCtType =
        dyn_cast<lwe::LWECiphertextType>(getElementTypeOrSelf(op.getInput()));
    int64_t targetLevelVal =
        outputCtType ? outputCtType.getModulusChain().getCurrent() : 0;

    if (!outputCtType || !inputCtType) {
      return op.emitOpError() << "expected ciphertext input and output types";
    }

    auto bucketKind =
        getCheddarLevelReduceBucketKind(op.getOperation(), outputCtType);
    if (failed(bucketKind)) {
      return op.emitOpError()
             << "unsupported CHEDDAR level_reduce scaling factor";
    }

    auto cheddarCtType =
        this->typeConverter->convertType(op.getOutput().getType());
    if (*bucketKind == CheddarLevelReduceBucketKind::kCanonical) {
      auto targetLevel = rewriter.getI64IntegerAttr(targetLevelVal);
      rewriter.replaceOpWithNewOp<cheddar::LevelDownOp>(
          op, cheddarCtType, ctx.value(), adaptor.getInput(), targetLevel);
      return success();
    }

    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    auto logDefaultScale = getCKKSLogDefaultScale(op.getOperation());
    if (failed(logDefaultScale)) {
      return op.emitOpError() << "missing CKKS scheme parameter";
    }

    int64_t currentLevel = inputCtType.getModulusChain().getCurrent();
    if (targetLevelVal < 0 || targetLevelVal > currentLevel) {
      return op.emitOpError()
             << "cannot level_reduce from level " << currentLevel
             << " to incompatible target level " << targetLevelVal;
    }
    Value current = adaptor.getInput();
    auto one = arith::ConstantOp::create(rewriter, op.getLoc(),
                                         rewriter.getF64FloatAttr(1.0));
    while (currentLevel > targetLevelVal) {
      current = cheddar::RescaleOp::create(rewriter, op.getLoc(), cheddarCtType,
                                           ctx.value(), current);
      --currentLevel;
      auto encodedOne = cheddar::EncodeConstantOp::create(
          rewriter, op.getLoc(), cheddar::ConstantType::get(getContext()),
          encoder.value(), one, rewriter.getI64IntegerAttr(currentLevel),
          rewriter.getI64IntegerAttr(*logDefaultScale));
      current =
          cheddar::MultConstOp::create(rewriter, op.getLoc(), cheddarCtType,
                                       ctx.value(), current, encodedOne);
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

// Bootstrap
struct ConvertCKKSBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    auto evkMap = cheddar::GetEvkMapOp::create(
        rewriter, op.getLoc(), cheddar::EvkMapType::get(getContext()),
        ui.value());

    rewriter.replaceOpWithNewOp<cheddar::BootOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), evkMap);
    return success();
  }
};

// Encode
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    auto invEncoding =
        dyn_cast<lwe::InverseCanonicalEncodingAttr>(op.getEncoding());
    if (!invEncoding) {
      return op.emitOpError()
             << "requires inverse-canonical CKKS plaintext encoding for "
                "CHEDDAR lowering";
    }
    // In HEIR's main LWE encoding, the scaling factor stored on
    // inverse_canonical_encoding is already log2-of-scale (additive for CKKS
    // multiplies), so it maps directly onto CHEDDAR's logScale field.
    int64_t logScale = invEncoding.getScalingFactor();

    // lwe.rlwe_encode doesn't carry a plaintext level on main, so encode at
    // the top of the modulus chain. Cross-level plaintexts (e.g. a mask used
    // before and after a rescale, or mult_plain with a ciphertext that has
    // already been rescaled) need proper level management and are not
    // supported yet — those tests should be kept out of CI until MGMT lands.
    int64_t level = 0;
    // When the chain is longer than the circuit needs (heir.level_offset > 0),
    // shift the level so CHEDDAR encodes plaintexts at the correct prime set.
    if (auto levelOffset =
            op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
                "heir.level_offset")) {
      level += levelOffset.getInt();
    }
    auto ptTy = cheddar::PlaintextType::get(getContext());
    rewriter.replaceOpWithNewOp<cheddar::EncodeOp>(
        op, ptTy, encoder.value(), adaptor.getInput(),
        rewriter.getI64IntegerAttr(level),
        rewriter.getI64IntegerAttr(logScale));
    return success();
  }
};

// Decrypt
struct ConvertLWEDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    rewriter.replaceOpWithNewOp<cheddar::DecryptOp>(
        op, cheddar::PlaintextType::get(getContext()), ui.value(),
        adaptor.getInput());
    return success();
  }
};

// Encrypt
struct ConvertLWEEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    rewriter.replaceOpWithNewOp<cheddar::EncryptOp>(
        op, cheddar::CiphertextType::get(getContext()), ui.value(),
        adaptor.getInput());
    return success();
  }
};

// Decode
struct ConvertLWEDecodeOp : public OpConversionPattern<lwe::RLWEDecodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    rewriter.replaceOpWithNewOp<cheddar::DecodeOp>(
        op, op.getOutput().getType(), encoder.value(), adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddEvaluatorArg pattern
//===----------------------------------------------------------------------===//

struct AddCheddarContextArg : public OpConversionPattern<func::FuncOp> {
  AddCheddarContextArg(
      mlir::MLIRContext* context,
      const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2),
        evaluators(evaluators) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "AddCheddarContextArg for func " << op.getName() << "\n");
    SmallVector<Type> selectedTypes =
        getRequiredCheddarContextTypes(op, evaluators);

    if (selectedTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "no CHEDDAR context needed");
    }
    if (hasLeadingTypes(op.getArgumentTypes(), selectedTypes)) {
      return rewriter.notifyMatchFailure(
          op, "CHEDDAR context arguments already present");
    }

    for (Type selectedType : selectedTypes) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Adding context arg of type: " << selectedType << "\n");
    }

    SmallVector<unsigned> argIndices(selectedTypes.size(), 0);
    SmallVector<DictionaryAttr> argAttrs(selectedTypes.size(), nullptr);
    SmallVector<Location> argLocs(selectedTypes.size(), op.getLoc());

    rewriter.modifyOpInPlace(op, [&] {
      SmallVector<unsigned> indices(selectedTypes.size(), 0);
      (void)op.insertArguments(indices, selectedTypes, argAttrs, argLocs);
    });
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

struct ConvertCheddarFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertCheddarFuncCallOp(
      const mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
      const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::CallOp>(typeConverter, context,
                                          /*benefit=*/2),
        evaluators(evaluators) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = getCalledFunction(op);
    if (failed(funcOp)) {
      return rewriter.notifyMatchFailure(op, "could not find callee function");
    }

    SmallVector<Type> requiredTypes =
        getRequiredCheddarContextTypes(funcOp.value(), evaluators);
    if (hasLeadingTypes(op.getOperandTypes(), requiredTypes) &&
        this->typeConverter->isLegal(op)) {
      return rewriter.notifyMatchFailure(
          op, "call already has required CHEDDAR signature");
    }

    SmallVector<Value> selectedValues;
    for (Type requiredType : requiredTypes) {
      auto result = getContextualArg(op.getOperation(), requiredType);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(op,
                                           "missing required CHEDDAR context");
      }
      selectedValues.push_back(result.value());
    }

    SmallVector<Value> newOperands;
    for (auto v : selectedValues) newOperands.push_back(v);
    for (auto operand : adaptor.getOperands()) newOperands.push_back(operand);

    SmallVector<Type> convertedResultTypes;
    if (failed(this->typeConverter->convertTypes(op.getResultTypes(),
                                                 convertedResultTypes))) {
      return rewriter.notifyMatchFailure(op,
                                         "could not convert call result types");
    }

    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, op.getCallee(),
                                          convertedResultTypes, newOperands)
        ->setDialectAttrs(dialectAttrs);
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LWETOCHEDDAR
#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h.inc"

struct LWEToCheddar : public impl::LWEToCheddarBase<LWEToCheddar> {
  // Workaround: dialect conversion may drop dialect attrs on func::CallOp.
  // Save before conversion and restore after.
  SmallVector<SmallVector<NamedAttribute>> funcCallOpDialectAttrs;

  void saveFuncCallOpDialectAttrs() {
    funcCallOpDialectAttrs.clear();
    auto* module = getOperation();
    module->walk([&](func::CallOp callOp) {
      SmallVector<NamedAttribute> dialectAttrs;
      for (auto namedAttr : callOp->getDialectAttrs()) {
        dialectAttrs.push_back(namedAttr);
      }
      funcCallOpDialectAttrs.push_back(dialectAttrs);
    });
  }

  void restoreFuncCallOpDialectAttrs() {
    auto* module = getOperation();
    auto* iter = funcCallOpDialectAttrs.begin();
    module->walk([&](func::CallOp callOp) {
      callOp->setDialectAttrs(*iter);
      ++iter;
    });
  }

  void runOnOperation() override {
    saveFuncCallOpDialectAttrs();

    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToCheddarTypeConverter typeConverter(context);

    // Only run for CKKS modules (CHEDDAR is CKKS-only)
    if (!moduleIsCKKS(module)) {
      module->emitOpError("CHEDDAR backend only supports CKKS scheme");
      return signalPassFailure();
    }

    ConversionTarget target(*context);
    target.addLegalDialect<cheddar::CheddarDialect>();
    target.addIllegalDialect<ckks::CKKSDialect>();
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::RAddPlainOp, lwe::RSubPlainOp, lwe::RMulPlainOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    addTensorConversionPatterns(typeConverter, patterns, target);

    // Predicate: function contains CKKS/LWE ops or operands. This must remain
    // true while dialect conversion is in flight, even after the generic func
    // signature conversion has already rewritten LWE-typed arguments.
    auto hasCryptoOps = [&](Operation* op) -> bool {
      return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(
                 op) ||
             containsDialects<lwe::LWEDialect, ckks::CKKSDialect>(op);
    };

    // Predicate: function contains encode ops.
    // Note: unlike Lattigo's containsEncode which uses walkFuncAndCallees to
    // transitively walk through call sites, we use a local walk here. This is
    // sufficient because AddCheddarContextArg processes each func independently
    // and ConvertCheddarFuncCallOp threads context args from caller to callee.
    // A caller that invokes a callee with encode ops will already have crypto
    // args (since encode ops accompany crypto ops in practice), so the Encoder
    // arg is threaded transitively via the hasCryptoOps predicate.
    auto hasEncodeOps = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](lwe::RLWEEncodeOp) { found = true; });
      return found;
    };

    // CHEDDAR context args to thread through functions
    std::vector<std::pair<Type, OpPredicate>> evaluators = {
        {cheddar::ContextType::get(context), hasCryptoOps},
        {cheddar::EncoderType::get(context),
         [&](Operation* op) { return hasCryptoOps(op) || hasEncodeOps(op); }},
        {cheddar::UserInterfaceType::get(context), hasCryptoOps},
    };

    // CKKS ops (scheme-specific ops that CKKSToLWE leaves in place)
    patterns.add<ConvertCKKSNegateOp, ConvertCKKSRelinOp, ConvertCKKSRescaleOp,
                 ConvertCKKSRotateOp, ConvertCKKSLevelReduceOp,
                 ConvertCKKSBootstrapOp>(typeConverter, context);

    // LWE R* ops (produced by torch-linalg-to-ckks pipeline)
    patterns.add<ConvertRAddOp, ConvertRSubOp, ConvertRMulOp,
                 ConvertRAddPlainOp, ConvertRSubPlainOp, ConvertRMulPlainOp>(
        typeConverter, context);

    // LWE encrypt/decrypt/encode/decode ops
    patterns.add<ConvertLWEEncodeOp, ConvertLWEDecodeOp, ConvertLWEEncryptOp,
                 ConvertLWEDecryptOp>(typeConverter, context);

    addRequiredCheddarContextArgs(cast<ModuleOp>(getOperation()), evaluators);
    if (failed(addRequiredCheddarContextOperandsToCalls(
            cast<ModuleOp>(getOperation()), evaluators))) {
      return signalPassFailure();
    }

    // Dynamically legal: func ops that have been converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      SmallVector<Type> requiredTypes =
          getRequiredCheddarContextTypes(op, evaluators);
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             hasLeadingTypes(op.getArgumentTypes(), requiredTypes);
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      FailureOr<func::FuncOp> callee = getCalledFunction(op);
      if (failed(callee)) {
        return false;
      }
      SmallVector<Type> requiredTypes =
          getRequiredCheddarContextTypes(callee.value(), evaluators);
      return typeConverter.isLegal(op) &&
             hasLeadingTypes(op.getCalleeType().getInputs(), requiredTypes);
    });

    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }

    restoreFuncCallOpDialectAttrs();
  }
};

}  // namespace mlir::heir::lwe

// Include the generated pass definition
// (must be after the struct definition for the base class to find it)
