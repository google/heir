#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizationDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/RotationUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "lwe-to-cheddar"

// TODO(#3429): lower linear_transform-related ops

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOCHEDDAR
#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h.inc"

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//
//
// The cheddar dialect is destination-passing-style on builtin tensors: a scalar
// payload value is a rank-0 `tensor<!cheddar.X>`. So a scalar `!lwe.ciphertext`
// converts to `tensor<!cheddar.ciphertext>` (rank-0), and a packed
// `tensor<Nx!lwe.ciphertext>` converts to `tensor<Nx!cheddar.ciphertext>` (a
// tensor whose ELEMENT is the scalar cheddar payload -- NOT a nested tensor, so
// the RankedTensorType rule maps payload elements directly rather than
// recursing through the scalar rule).

class ToCheddarTypeConverter : public TypeConverter {
 public:
  explicit ToCheddarTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      return RankedTensorType::get({}, cheddar::CiphertextType::get(ctx));
    });
    addConversion([ctx](lwe::LWEPlaintextType type) -> Type {
      return RankedTensorType::get({}, cheddar::PlaintextType::get(ctx));
    });
    // Keys are absorbed into the UserInterface (threaded as contextual args).
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([this, ctx](RankedTensorType type) -> Type {
      Type elt = type.getElementType();
      // A packed payload buffer maps to a tensor of the SCALAR cheddar payload
      // (the scalar-payload rules above map the bare element type to a rank-0
      // tensor, which must not be nested inside this one).
      if (isa<lwe::LWECiphertextType>(elt))
        return RankedTensorType::get(type.getShape(),
                                     cheddar::CiphertextType::get(ctx));
      if (isa<lwe::LWEPlaintextType>(elt))
        return RankedTensorType::get(type.getShape(),
                                     cheddar::PlaintextType::get(ctx));
      return RankedTensorType::get(type.getShape(), this->convertType(elt));
    });
    // split-preprocessing storage: convert its plaintext element types
    // (lwe.lwe_plaintext -> rank-0 tensor<!cheddar.plaintext>); the storage
    // itself is lowered to memref later by --preprocessing-to-cheddar.
    addConversion([this](preprocessing::PreprocessingStorageType type) -> Type {
      return preprocessing::convertStorageElementTypes(type, this);
    });
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

bool isCheddarPayload(Type t) {
  return isa<cheddar::CiphertextType, cheddar::PlaintextType,
             cheddar::ConstantType>(t);
}

// A shape-only destination for operations without a reusable payload input.
// One-Shot Bufferize chooses the eventual allocation.
Value makeEmptyDest(OpBuilder& b, Location loc, Type resultTy) {
  auto tensorType = cast<RankedTensorType>(resultTy);
  return tensor::EmptyOp::create(b, loc, tensorType.getShape(),
                                 tensorType.getElementType());
}

// Reuse only a payload produced locally by another Cheddar operation. All
// scale-snu APIs represented by Cheddar DPS ops accept an explicitly identical
// input and output object, but a function argument or preprocessing-storage
// view is borrowed by the generated C++ ABI and must not become mutable merely
// because it happens to die in MLIR SSA. One-Shot Bufferize still resolves
// later-use conflicts for eligible local values by allocating an uninitialized
// buffer for the fully-overwriting op.
Value makeReusableDest(OpBuilder& b, Location loc, Type resultTy,
                       Value candidate) {
  if (candidate.getType() == resultTy) {
    Operation* definingOp = candidate.getDefiningOp();
    if (definingOp && definingOp->getDialect() &&
        definingOp->getDialect()->getNamespace() ==
            cheddar::CheddarDialect::getDialectNamespace())
      return candidate;
  }
  return makeEmptyDest(b, loc, resultTy);
}

template <typename CheddarType>
FailureOr<Value> getContextualArg(Operation* op) {
  auto result = getContextualArgFromFunc<CheddarType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found op in a function without a required CHEDDAR context "
              "argument. Did the AddCheddarContextArg pattern fail to run?";
  }
  return result.value();
}

FailureOr<Value> getContextualContext(Operation* op) {
  if (auto bootCtx = getContextualArgFromFunc<cheddar::BootContextType>(op);
      succeeded(bootCtx))
    return bootCtx;
  return getContextualArg<cheddar::ContextType>(op);
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
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    // CHEDDAR's element-wise arithmetic APIs can overwrite their lhs. This is
    // only a DPS destination hint: One-Shot Bufferize still allocates a fresh
    // output if lhs remains live.
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getLhs());
    auto result =
        CheddarOp::create(rewriter, op.getLoc(), resultTy, ctx.value(),
                          adaptor.getLhs(), adaptor.getRhs(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

using ConvertCKKSAddOp = ConvertCKKSBinOp<ckks::AddOp, cheddar::AddOp>;
using ConvertCKKSSubOp = ConvertCKKSBinOp<ckks::SubOp, cheddar::SubOp>;
using ConvertCKKSMulOp = ConvertCKKSBinOp<ckks::MulOp, cheddar::MultOp>;
using ConvertRAddOp = ConvertCKKSBinOp<lwe::RAddOp, cheddar::AddOp>;
using ConvertRSubOp = ConvertCKKSBinOp<lwe::RSubOp, cheddar::SubOp>;
using ConvertRMulOp = ConvertCKKSBinOp<lwe::RMulOp, cheddar::MultOp>;

// Ct-pt operations.
template <typename CKKSOp, typename CheddarOp>
struct ConvertCKKSPlainOp : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    // Ensure ciphertext is first operand (CHEDDAR convention).
    auto isCt = [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && isa<cheddar::CiphertextType>(t.getElementType());
    };
    Value ciphertext = adaptor.getLhs();
    Value plaintext = adaptor.getRhs();
    if (!isCt(ciphertext)) {
      ciphertext = adaptor.getRhs();
      plaintext = adaptor.getLhs();
    }
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    // These APIs can overwrite the ciphertext operand. One-Shot Bufferize
    // decides whether doing so is valid for this particular SSA use-def chain.
    Value dest = makeReusableDest(rewriter, op.getLoc(), resultTy, ciphertext);
    auto result = CheddarOp::create(rewriter, op.getLoc(), resultTy,
                                    ctx.value(), ciphertext, plaintext, dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Ct-pt subtraction. cheddar.sub_plain computes ct - pt, so it requires the
// ciphertext first -- but unlike add/mul, subtraction is NOT commutative, so a
// blind operand swap turns `pt - ct` into `ct - pt` (a sign flip). When the
// plaintext is the lhs, lower `pt - ct` as `(-ct) + pt` (negate then
// add_plain). Mirrors LWEToLattigo's ConvertRlweSubPlainOp.
template <typename CKKSOp>
struct ConvertCKKSSubPlainOpImpl : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto isCt = [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && isa<cheddar::CiphertextType>(t.getElementType());
    };
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    if (isCt(adaptor.getLhs())) {
      // ct - pt: direct.
      Value dest =
          makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getLhs());
      auto result = cheddar::SubPlainOp::create(rewriter, op.getLoc(), resultTy,
                                                ctx.value(), adaptor.getLhs(),
                                                adaptor.getRhs(), dest);
      rewriter.replaceOp(op, result);
      return success();
    }
    // pt - ct  ==  (-ct) + pt
    Value plaintext = adaptor.getLhs();
    Value ciphertext = adaptor.getRhs();
    Value negDest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, ciphertext);
    Value negated = cheddar::NegOp::create(rewriter, op.getLoc(), resultTy,
                                           ctx.value(), ciphertext, negDest)
                        ->getResult(0);
    Value addDest = negated;
    auto result =
        cheddar::AddPlainOp::create(rewriter, op.getLoc(), resultTy,
                                    ctx.value(), negated, plaintext, addDest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

using ConvertCKKSAddPlainOp =
    ConvertCKKSPlainOp<ckks::AddPlainOp, cheddar::AddPlainOp>;
using ConvertCKKSSubPlainOp = ConvertCKKSSubPlainOpImpl<ckks::SubPlainOp>;
using ConvertCKKSMulPlainOp =
    ConvertCKKSPlainOp<ckks::MulPlainOp, cheddar::MultPlainOp>;
using ConvertRAddPlainOp =
    ConvertCKKSPlainOp<lwe::RAddPlainOp, cheddar::AddPlainOp>;
using ConvertRSubPlainOp = ConvertCKKSSubPlainOpImpl<lwe::RSubPlainOp>;
using ConvertRMulPlainOp =
    ConvertCKKSPlainOp<lwe::RMulPlainOp, cheddar::MultPlainOp>;

template <typename SourceOp>
struct ConvertNegateOp : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto negated = cheddar::NegOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, negated);
    return success();
  }
};

using ConvertCKKSNegateOp = ConvertNegateOp<ckks::NegateOp>;
using ConvertRNegateOp = ConvertNegateOp<lwe::RNegateOp>;

struct ConvertCKKSRelinOp : public OpConversionPattern<ckks::RelinearizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto multKey = getContextualArg<cheddar::EvalKeyType>(op.getOperation());
    if (failed(multKey)) return multKey;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result = cheddar::RelinearizeOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(),
        multKey.value(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertCKKSRescaleOp : public OpConversionPattern<ckks::RescaleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RescaleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result = cheddar::RescaleOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertCKKSRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evk = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evk)) return evk;

    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift)
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");

    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    auto inputType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getInput().getType()));
    if (!inputType || !inputType.getModulusChain())
      return op.emitOpError(
          "cannot determine the ciphertext level for rotation");

    IntegerAttr level =
        rewriter.getI64IntegerAttr(inputType.getModulusChain().getCurrent());

    if (staticShift) {
      auto polyMod =
          inputType.getPlaintextSpace().getRing().getPolynomialModulus();
      if (!polyMod)
        return op.emitOpError(
            "ciphertext plaintext space ring has no polynomial modulus");
      int64_t ringDegree = polyMod.getPolynomial().getDegree();
      int64_t distance =
          normalizeRotation(staticShift.getInt(), ringDegree / 2);
      if (distance == 0) {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
      }
      staticShift = rewriter.getI64IntegerAttr(distance);
    }

    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    if (dynamicShift) {
      auto result = cheddar::HRotOp::create(
          rewriter, op.getLoc(), resultTy, ctx.value(), evk.value(),
          adaptor.getInput(), dest, dynamicShift,
          /*static_distance=*/IntegerAttr(), level);
      rewriter.replaceOp(op, result);
    } else {
      auto result = cheddar::HRotOp::create(
          rewriter, op.getLoc(), resultTy, ctx.value(), evk.value(),
          adaptor.getInput(), dest,
          /*dynamic_distance=*/Value(), staticShift, level);
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct ConvertCKKSLevelReduceOp
    : public OpConversionPattern<ckks::LevelReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::LevelReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto outputCtType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getOutput().getType()));
    if (!outputCtType || !outputCtType.getModulusChain())
      return op.emitOpError(
          "cannot lower level_reduce without an output modulus chain");
    int64_t targetLevelVal = outputCtType.getModulusChain().getCurrent();
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result = cheddar::LevelDownOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(), dest,
        rewriter.getI64IntegerAttr(targetLevelVal));
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertCKKSBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getTargetLevel())
      return rewriter.notifyMatchFailure(
          op,
          "bootstrap target levels must be resolved before CHEDDAR lowering");
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result =
        cheddar::BootOp::create(rewriter, op.getLoc(), resultTy, ctx.value(),
                                adaptor.getInput(), evkMap.value(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Encode at the level and logarithmic scale chosen by the upstream CKKS scale
// management pipeline. CHEDDAR accepts the corresponding linear scale.
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;
    if (!op.getLevel())
      return op.emitOpError()
             << "cannot lower to cheddar.encode without an explicit level";
    int64_t level = op.getLevel().value();
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultTy);
    auto result = cheddar::EncodeOp::create(
        rewriter, op.getLoc(), resultTy, encoder.value(), adaptor.getInput(),
        dest, rewriter.getI64IntegerAttr(level), op.getScaleAttr());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLWEDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultTy);
    auto result = cheddar::DecryptOp::create(
        rewriter, op.getLoc(), resultTy, ui.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLWEEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultTy);
    auto result = cheddar::EncryptOp::create(
        rewriter, op.getLoc(), resultTy, ui.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Decode is already destination-passing on the float `value` buffer.
struct ConvertLWEDecodeOp : public OpConversionPattern<lwe::RLWEDecodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEDecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;
    auto outTy = cast<RankedTensorType>(op.getOutput().getType());
    Value dest = tensor::EmptyOp::create(
        rewriter, op.getLoc(), outTy.getShape(), outTy.getElementType());
    auto result =
        cheddar::DecodeOp::create(rewriter, op.getLoc(), outTy, encoder.value(),
                                  adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Payload packing: scalar-index tensor ops -> rank-reducing slice ops
//===----------------------------------------------------------------------===//
//
// In DPS form a "scalar" payload is a rank-0 `tensor<!cheddar.X>`, so the
// source's scalar packing ops (which produce / consume the bare payload element
// of a `tensor<Nx!lwe.X>`) must become rank-reducing slice ops:
//   tensor.extract %v[i]          -> tensor.extract_slice %v[i][1][1] : ->
//   tensor<!X> tensor.insert  %s into %v[i]  -> tensor.insert_slice  %s into
//   %v[i][1][1] tensor.from_elements %s0,..   -> tensor.empty + insert_slice
//   per element

static void unitSlice(OpBuilder& b, ValueRange indices,
                      SmallVector<OpFoldResult>& offsets,
                      SmallVector<OpFoldResult>& sizes,
                      SmallVector<OpFoldResult>& strides) {
  for (Value idx : indices) {
    offsets.push_back(idx);
    sizes.push_back(b.getIndexAttr(1));
    strides.push_back(b.getIndexAttr(1));
  }
}

struct ConvertPayloadExtract : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(adaptor.getTensor().getType());
    if (!srcTy || !isCheddarPayload(srcTy.getElementType())) return failure();
    auto resTy = RankedTensorType::get({}, srcTy.getElementType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    unitSlice(rewriter, adaptor.getIndices(), offsets, sizes, strides);
    auto result = tensor::ExtractSliceOp::create(rewriter, op.getLoc(), resTy,
                                                 adaptor.getTensor(), offsets,
                                                 sizes, strides);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertPayloadInsert : public OpConversionPattern<tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto destTy = dyn_cast<RankedTensorType>(adaptor.getDest().getType());
    if (!destTy || !isCheddarPayload(destTy.getElementType())) return failure();
    SmallVector<OpFoldResult> offsets, sizes, strides;
    unitSlice(rewriter, adaptor.getIndices(), offsets, sizes, strides);
    auto result = tensor::InsertSliceOp::create(
        rewriter, op.getLoc(), adaptor.getScalar(), adaptor.getDest(), offsets,
        sizes, strides);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertPayloadFromElements
    : public OpConversionPattern<tensor::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::FromElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resTy =
        dyn_cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    if (!resTy || !isCheddarPayload(resTy.getElementType())) return failure();
    if (resTy.getRank() == 0 && adaptor.getElements().size() == 1) {
      rewriter.replaceOp(op, adaptor.getElements().front());
      return success();
    }
    if (resTy.getRank() != 1 ||
        resTy.getDimSize(0) != (int64_t)adaptor.getElements().size())
      return rewriter.notifyMatchFailure(op, "unsupported from_elements shape");
    Value acc = tensor::EmptyOp::create(rewriter, op.getLoc(), resTy.getShape(),
                                        resTy.getElementType());
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    for (auto [i, elt] : llvm::enumerate(adaptor.getElements())) {
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr((int64_t)i)};
      acc = tensor::InsertSliceOp::create(rewriter, op.getLoc(), elt, acc,
                                          offsets, sizes, strides);
    }
    rewriter.replaceOp(op, acc);
    return success();
  }
};

// A bare LWE payload converts to a rank-0 tensor in Cheddar's DPS form.  When
// tensor.splat packs such a payload, extract the Cheddar scalar first; the
// generic structural conversion would otherwise feed tensor<!cheddar.X> to an
// op whose operand must be !cheddar.X.
struct ConvertPayloadSplat : public OpConversionPattern<tensor::SplatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::SplatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto inputTy = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultTy =
        dyn_cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    if (!inputTy || inputTy.getRank() != 0 ||
        !isCheddarPayload(inputTy.getElementType()) || !resultTy ||
        !isCheddarPayload(resultTy.getElementType()))
      return failure();

    auto scalar = tensor::ExtractOp::create(rewriter, op.getLoc(),
                                            adaptor.getInput(), ValueRange{});
    auto splat = tensor::SplatOp::create(rewriter, op.getLoc(), resultTy,
                                         scalar, ValueRange{});
    rewriter.replaceOp(op, splat.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Context-argument threading (mirrors LWEToLattigo)
//===----------------------------------------------------------------------===//

using SupportRequirements = DenseMap<func::FuncOp, DenseSet<Type>>;

// Walk the IR and find, for each function, the set of types needed to supply as
// function arguments. E.g., a function that contains a bootstrap op needs a
// BootContext argument supplied to it.
SupportRequirements collectSupportRequirements(Operation* module) {
  auto* context = module->getContext();
  Type ctxType = cheddar::ContextType::get(context);
  Type bootType = cheddar::BootContextType::get(context);
  Type encoderType = cheddar::EncoderType::get(context);
  Type uiType = cheddar::UserInterfaceType::get(context);
  Type keyType = cheddar::EvalKeyType::get(context);
  Type mapType = cheddar::EvkMapType::get(context);

  SupportRequirements required;
  module->walk([&](func::FuncOp function) {
    auto& types = required[function];
    walkFuncAndCallees(function, [&](Operation* op) {
      if (isa<ckks::AddOp, ckks::SubOp, ckks::MulOp, ckks::AddPlainOp,
              ckks::SubPlainOp, ckks::MulPlainOp, ckks::NegateOp,
              ckks::RelinearizeOp, ckks::RescaleOp, ckks::RotateOp,
              ckks::LevelReduceOp, ckks::BootstrapOp, lwe::RAddOp, lwe::RSubOp,
              lwe::RMulOp, lwe::RNegateOp, lwe::RAddPlainOp, lwe::RSubPlainOp,
              lwe::RMulPlainOp>(op))
        types.insert(ctxType);
      if (isa<ckks::BootstrapOp>(op)) types.insert(bootType);
      if (isa<lwe::RLWEEncodeOp, lwe::RLWEDecodeOp>(op))
        types.insert(encoderType);
      if (isa<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp>(op)) types.insert(uiType);
      if (isa<ckks::RelinearizeOp>(op)) types.insert(keyType);
      if (isa<ckks::RotateOp, ckks::BootstrapOp>(op)) types.insert(mapType);
      if (auto call = dyn_cast<func::CallOp>(op);
          call && isDebugPort(call.getCallee())) {
        types.insert(encoderType);
        types.insert(uiType);
      }
      return WalkResult::advance();
    });
  });
  return required;
}

// This list defines a unique ordering of support arguments, so that patterns
// that insert or modify function signatures or calls can ensure they're putting
// the (unique) instances of each value/type in a consistent order.
inline DenseMap<Type, int> getFixedOrderOfSupportTypes(MLIRContext* ctx) {
  return {{cheddar::ContextType::get(ctx), 0},
          {cheddar::BootContextType::get(ctx), 1},
          {cheddar::EncoderType::get(ctx), 2},
          {cheddar::UserInterfaceType::get(ctx), 3},
          {cheddar::EvalKeyType::get(ctx), 4},
          {cheddar::EvkMapType::get(ctx), 5}};
}

void sortSupportValues(SmallVector<Value>& values, MLIRContext* ctx) {
  DenseMap<Type, int> typeToIndex = getFixedOrderOfSupportTypes(ctx);
  llvm::sort(values, [&](Value v1, Value v2) {
    return typeToIndex[v1.getType()] < typeToIndex[v2.getType()];
  });
}

void sortSupportTypes(SmallVector<Type>& types, MLIRContext* ctx) {
  DenseMap<Type, int> typeToIndex = getFixedOrderOfSupportTypes(ctx);
  llvm::sort(types, [&](Type t1, Type t2) {
    return typeToIndex[t1] < typeToIndex[t2];
  });
}

struct AddCheddarContextArg : public OpConversionPattern<func::FuncOp> {
  AddCheddarContextArg(const TypeConverter& converter,
                       mlir::MLIRContext* context,
                       const SupportRequirements& required)
      : OpConversionPattern<func::FuncOp>(converter, context,
                                          /* benefit= */ 2),
        required(required) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> selectedTypes;
    for (Type supportType : required.lookup(op)) {
      // Skip a context type already present as an argument: with --debug
      // enabled the lwe debug port adds an LWESecretKey arg that converts to a
      // UserInterface, which would otherwise be duplicated here (and break
      // call-site threading, which dedupes by type).
      if (llvm::any_of(op.getArgumentTypes(), [&](Type type) {
            return getTypeConverter()->convertType(type) == supportType;
          }))
        continue;
      selectedTypes.push_back(supportType);
    }
    if (selectedTypes.empty())
      return rewriter.notifyMatchFailure(op, "no CHEDDAR context needed");

    sortSupportTypes(selectedTypes, op->getContext());

    SmallVector<DictionaryAttr> argAttrs(selectedTypes.size(), nullptr);
    SmallVector<Location> argLocs(selectedTypes.size(), op.getLoc());
    rewriter.modifyOpInPlace(op, [&] {
      SmallVector<unsigned> indices(selectedTypes.size(), 0);
      (void)op.insertArguments(indices, selectedTypes, argAttrs, argLocs);
    });
    return success();
  }

 private:
  const SupportRequirements& required;
};

struct ConvertCheddarFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertCheddarFuncCallOp(const TypeConverter& converter,
                           mlir::MLIRContext* context,
                           const SupportRequirements& required)
      : OpConversionPattern<func::CallOp>(converter, context),
        required(required) {}

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = getCalledFunction(op);
    if (failed(funcOp))
      return rewriter.notifyMatchFailure(op, "could not find callee function");
    SmallVector<Value> newOperands;
    // Each support type required by the called func must be threaded through
    // the current func.
    for (Type supportType : required.lookup(*funcOp)) {
      auto result = getContextualArgFromFunc(op.getOperation(), supportType);
      if (failed(result)) {
        return op.emitOpError() << "requires support type " << supportType
                                << " but the func containing this call op has "
                                   "no such argument to thread through.";
      };

      // Skip if the func.call op already passes the needed arg.
      if (llvm::any_of(adaptor.getOperands(), [&](Value operand) {
            return operand.getType() == supportType;
          }))
        continue;
      newOperands.push_back(result.value());
    }

    sortSupportValues(newOperands, op->getContext());
    llvm::append_range(newOperands, adaptor.getOperands());

    // The callee's result types are type-converted by the structural func
    // pattern (e.g. tensor<Nx!lwe.plaintext> -> tensor<Nx!cheddar.plaintext>),
    // so the rebuilt call must use the converted result types or it stays
    // signature-inconsistent with its callee and fails to legalize. Operand
    // types come pre-converted via the adaptor.
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     newResultTypes, newOperands);
    call->setDialectAttrs(dialectAttrs);
    rewriter.replaceOp(op, call);
    return success();
  }

 private:
  const SupportRequirements& required;
};

//===----------------------------------------------------------------------===//
// Debug port (__heir_debug_*) handling
//===----------------------------------------------------------------------===//
//
// `lwe-add-debug-port` lowers each `debug.validate` to a call to an external
// `func.func private @__heir_debug_N(%key: !lwe.secret_key, %ct:
// !lwe.ciphertext)`. To decrypt AND decode the ciphertext for printing, the
// CHEDDAR-side hook needs an `Encoder` (decode) and a `UserInterface`
// (decrypt), so we re-shape both the external declaration and every call site
// to
// `(Encoder, UserInterface, Ciphertext)`. The original `!lwe.secret_key`
// operand (which converts to a UserInterface) is dropped in favour of the
// contextual UserInterface threaded into the enclosing function, matching how
// all other CHEDDAR ops obtain their context args. The CheddarToEmitC pass then
// emits these calls as `__heir_debug(...)` C++ calls.

// The external __heir_debug_* declaration: reshape its signature to
// (Encoder, UserInterface, <converted ciphertext>) -> (). The ciphertext is the
// original last operand (a scalar `!lwe.ciphertext` -> rank-0
// `tensor<!cheddar.ciphertext>`, or a packed `tensor<Nx!lwe.ciphertext>` ->
// `tensor<Nx!cheddar.ciphertext>`), converted via the type converter.
struct ConvertDebugFuncDecl : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.isExternal() || !isDebugPort(op.getName())) return failure();
    auto inputs = op.getFunctionType().getInputs();
    if (inputs.empty()) return failure();
    Type ctType = typeConverter->convertType(inputs.back());
    if (!ctType) return failure();
    auto* ctx = getContext();
    SmallVector<Type> argTypes{cheddar::EncoderType::get(ctx),
                               cheddar::UserInterfaceType::get(ctx), ctType};
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(FunctionType::get(ctx, argTypes, {}));
      // __heir_debug only READS the ciphertext (decrypt+decode for printing).
      // Mark the ciphertext arg read-only so bufferization borrows it.
      // Without this, one-shot-bufferize treats the external call's operand
      // conservatively as possibly-written and materializes a copy -- which for
      // a move-only cheddar Ciphertext becomes a destructive std::move, leaving
      // the observed value (and its later uses) empty -> "num primes mismatch".
      op.setArgAttr(2, "bufferization.access", rewriter.getStringAttr("read"));
    });
    return success();
  }
};

// A call to __heir_debug_*: thread the enclosing function's Encoder +
// UserInterface contextual args, followed by the (converted) ciphertext operand
// (the last original operand; the original secret-key operand is dropped).
struct ConvertDebugCall : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!isDebugPort(op.getCallee())) return failure();
    auto* ctx = getContext();
    auto encoder = getContextualArgFromFunc(op.getOperation(),
                                            cheddar::EncoderType::get(ctx));
    if (failed(encoder)) return failure();
    auto ui = getContextualArgFromFunc(op.getOperation(),
                                       cheddar::UserInterfaceType::get(ctx));
    if (failed(ui)) return failure();
    SmallVector<Value> newOperands{encoder.value(), ui.value(),
                                   adaptor.getOperands().back()};
    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     TypeRange{}, newOperands);
    call->setDialectAttrs(dialectAttrs);
    rewriter.replaceOp(op, call);
    return success();
  }
};

void increaseAttrToAtLeast(Operation* module, StringRef attrName,
                           int minValue) {
  auto existingAttr = dyn_cast_or_null<IntegerAttr>(module->getAttr(attrName));
  int existingValue = existingAttr == nullptr ? 0 : existingAttr.getInt();
  module->setAttr(attrName,
                  IntegerAttr::get(IntegerType::get(module->getContext(), 64),
                                   std::max(existingValue, minValue)));
}

}  // namespace

struct LWEToCheddar : public impl::LWEToCheddarBase<LWEToCheddar> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToCheddarTypeConverter typeConverter(context);

    if (!moduleIsCKKS(module)) {
      module->emitOpError("CHEDDAR backend only supports CKKS scheme");
      return signalPassFailure();
    }

    // scale-snu/cheddar requires at least 256 slots in its special-FFT
    // bootstrap implementation.
    increaseAttrToAtLeast(module, kActualSlotCountAttrName, 256);
    increaseAttrToAtLeast(module, kRequestedSlotCountAttrName, 256);

    ConversionTarget target(*context);
    target.addLegalDialect<cheddar::CheddarDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addIllegalDialect<ckks::CKKSDialect, lwe::LWEDialect>();
    // preprocessing.* ops are legal once their plaintext element types have
    // been converted to cheddar's; --preprocessing-to-cheddar lowers them
    // after.
    target.addDynamicallyLegalDialect<preprocessing::PreprocessingDialect>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    addTensorConversionPatterns(typeConverter, patterns, target);
    preprocessing::populatePreprocessingConversions(patterns, typeConverter,
                                                    context);

    const auto required = collectSupportRequirements(module);
    patterns.add<AddCheddarContextArg>(typeConverter, context, required);
    patterns.add<ConvertCheddarFuncCallOp>(typeConverter, context, required);

    // Debug ports get dedicated, higher-benefit handling (the generic call /
    // structural func patterns would mis-thread their context args).
    patterns.add<ConvertDebugFuncDecl, ConvertDebugCall>(typeConverter, context,
                                                         /*benefit=*/3);

    patterns
        .add<ConvertCKKSAddOp, ConvertCKKSSubOp, ConvertCKKSMulOp,
             ConvertCKKSAddPlainOp, ConvertCKKSSubPlainOp,
             ConvertCKKSMulPlainOp, ConvertCKKSNegateOp, ConvertCKKSRelinOp,
             ConvertCKKSRescaleOp, ConvertCKKSRotateOp,
             ConvertCKKSLevelReduceOp, ConvertCKKSBootstrapOp, ConvertRAddOp,
             ConvertRSubOp, ConvertRMulOp, ConvertRNegateOp, ConvertRAddPlainOp,
             ConvertRSubPlainOp, ConvertRMulPlainOp, ConvertLWEEncodeOp,
             ConvertLWEDecodeOp, ConvertLWEEncryptOp, ConvertLWEDecryptOp>(
            typeConverter, context);

    // Payload packing ops -> rank-reducing slice ops (benefit 2 so they win
    // over the structural tensor conversion for payload-typed tensors).
    patterns.add<ConvertPayloadExtract, ConvertPayloadInsert,
                 ConvertPayloadFromElements, ConvertPayloadSplat>(
        typeConverter, context, /*benefit=*/2);

    // A reshaped `__heir_debug_*` declaration / call has exactly
    // (Encoder, UserInterface, tensor<...x!cheddar.ciphertext>) inputs and no
    // results.
    auto isReshapedDebugSig = [](TypeRange ins) {
      if (ins.size() != 3 || !isa<cheddar::EncoderType>(ins[0]) ||
          !isa<cheddar::UserInterfaceType>(ins[1]))
        return false;
      auto t = dyn_cast<RankedTensorType>(ins[2]);
      return t && isa<cheddar::CiphertextType>(t.getElementType());
    };
    auto isReshapedDebugDecl = [&](func::FuncOp op) {
      return op.getFunctionType().getNumResults() == 0 &&
             isReshapedDebugSig(op.getFunctionType().getInputs());
    };
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      if (isDebugPort(op.getName())) return isReshapedDebugDecl(op);
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             llvm::all_of(required.lookup(op), [&](Type type) {
               return llvm::is_contained(op.getArgumentTypes(), type);
             });
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      if (isDebugPort(op.getCallee()))
        return isReshapedDebugSig(op.getCalleeType().getInputs());
      auto callee = getCalledFunction(op);
      return succeeded(callee) && typeConverter.isLegal(op) &&
             callee->getFunctionType() == op.getCalleeType();
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

    // Name every support argument, including ones a function already carried
    // before conversion, so later stages read the role off the argument.
    module->walk([&](func::FuncOp function) {
      for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
        StringRef kind = cheddar::getSupportKind(type);
        if (kind.empty() ||
            function.getArgAttr(index, cheddar::kSupportArgAttrName))
          continue;
        function.setArgAttr(index, cheddar::kSupportArgAttrName,
                            StringAttr::get(context, kind));
      }
    });
  }
};

}  // namespace mlir::heir::lwe
