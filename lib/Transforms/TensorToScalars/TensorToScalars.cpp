#include "lib/Transforms/TensorToScalars/TensorToScalars.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Patterns.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_TENSORTOSCALARS
#include "lib/Transforms/TensorToScalars/TensorToScalars.h.inc"

namespace {

SmallVector<SmallVector<int64_t>> getAllIndices(TensorType tensorType) {
  auto getIndices = [&](ArrayRef<int64_t> shape) {
    SmallVector<SmallVector<int64_t>> oldResult;
    while (!shape.empty()) {
      SmallVector<SmallVector<int64_t>> result;
      for (auto i = 0; i < shape.front(); ++i) {
        if (oldResult.empty()) {
          result.push_back({i});
        }
        for (const auto& res : oldResult) {
          SmallVector<int64_t> newResult = res;
          newResult.push_back(i);
          result.push_back(newResult);
        }
      }
      shape = shape.drop_front();
      oldResult = result;
    }
    return oldResult;
  };
  return getIndices(tensorType.getShape());
}

}  // namespace

static Value buildFromElementsOp(OpBuilder& builder,
                                 RankedTensorType resultType, ValueRange inputs,
                                 Location loc) {
  return tensor::FromElementsOp::create(builder, loc, resultType, inputs);
}

static SmallVector<Value> buildExtractOps(OpBuilder& builder,
                                          TypeRange resultTypes,
                                          ValueRange inputs, Location loc) {
  // This pass only operates on tensors of static shape
  if (inputs.size() != 1) return {};
  Value input = inputs.front();
  RankedTensorType inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape()) return {};

  // Create extract ops in "natural" order (dimension-by-dimension)
  SmallVector<Value> values;
  ImplicitLocOpBuilder b(loc, builder);
  const auto* maxDimension = llvm::max_element(inputType.getShape());
  SmallVector<Value> constVals;
  for (auto i = 0; i < *maxDimension; ++i) {
    constVals.push_back(arith::ConstantIndexOp::create(b, i));
  }
  auto rawIndices = getAllIndices(inputType);
  for (SmallVector<int64_t> indices : rawIndices) {
    SmallVector<Value> idxRange = llvm::to_vector(
        llvm::map_range(indices, [&](int64_t idx) { return constVals[idx]; }));
    Value element = tensor::ExtractOp::create(builder, loc, input, idxRange);
    values.push_back(element);
  }
  return values;
}

class ConvertFromElementsOp
    : public OpConversionPattern<tensor::FromElementsOp> {
 public:
  using OpConversionPattern<tensor::FromElementsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::FromElementsOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Conversion has no Conversion-level illegality handling
    if (typeConverter->isLegal(op.getOperation())) return failure();
    // This pass only operates on tensors of static shape,
    // but no check is necessary here as from_elements' shape is always static
    // Replace the current op with the flattened operands.
    // This should already match the "natural" order expected by this pass.
    rewriter.replaceOpWithMultiple(op, adaptor.getOperands());
    return success();
  }
};

class ConvertInsertOp : public OpConversionPattern<tensor::InsertOp> {
 public:
  using OpConversionPattern<tensor::InsertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Conversion has no Conversion-level illegality handling
    if (typeConverter->isLegal(op.getOperation())) return failure();

    // This pass only operates on tensors of static shape
    if (!op.getResult().getType().hasStaticShape()) return failure();

    // Compute the insertion offset (in dimension-by-dimension order):
    int64_t multiplier = 1;
    int64_t offset = 0;
    for (auto [dim, idx] :
         llvm::zip(op.getResult().getType().getShape(), op.getIndices())) {
      // We can only support statically known indices
      // that have been constant-folded to a single arith.constant op
      auto cidx = idx.getDefiningOp<arith::ConstantIndexOp>();
      if (!cidx) return failure();
      offset += cidx.value() * multiplier;
      multiplier *= dim;
    }

    // get converted "tensor" operand from op (likely a unrealized_builtin_cast)
    SmallVector<Value> elements = adaptor.getOperands()[1];
    // replace element at offset with the "scalar" operand to be inserted
    elements[offset] = adaptor.getOperands()[0].front();
    // replace the current op with the converted operands.
    rewriter.replaceOpWithMultiple(op, {elements});
    return success();
  }
};

class ConvertExtractOp : public OpConversionPattern<tensor::ExtractOp> {
 public:
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Conversion has no Conversion-level illegality handling
    if (typeConverter->isLegal(op.getOperation())) return failure();

    // This pass only operates on tensors of static shape
    if (!op.getTensor().getType().hasStaticShape()) return failure();

    // Compute the extraction offset (in dimension-by-dimension order):
    int64_t multiplier = 1;
    int64_t offset = 0;
    SmallVector<OpFoldResult> indices = getAsOpFoldResult(op.getIndices());

    for (auto [dim, idx] :
         llvm::zip(op.getTensor().getType().getShape(), indices)) {
      // We can only support statically known indices
      // that have been constant-folded to a single arith.constant op
      auto constantIdx = dyn_cast<Attribute>(idx);
      if (!constantIdx) return failure();
      offset += cast<IntegerAttr>(constantIdx).getInt() * multiplier;
      multiplier *= dim;
    }

    // replace the current op with the extracted scalar.
    Value value = adaptor.getOperands()[0][offset];
    rewriter.replaceOpWithMultiple(op, {value});

    return success();
  }
};

class ConvertConstantOp : public OpConversionPattern<arith::ConstantOp> {
 public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ConstantOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Conversion has no Conversion-level illegality handling
    if (typeConverter->isLegal(op.getOperation())) return failure();

    RankedTensorType tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType) {
      return failure();
    }

    // This pass only operates on tensors of static shape
    if (!tensorType.hasStaticShape()) return failure();

    auto elementsAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr());
    if (!elementsAttr) return failure();

    // Replace the current op with flattened constants.
    SmallVector<Value> constants;
    for (auto valueAttr : elementsAttr.getValues<Attribute>()) {
      constants.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), elementsAttr.getElementType(),
          cast<TypedAttr>(valueAttr)));
    }

    rewriter.replaceOpWithMultiple(op, {constants});
    return success();
  }
};

struct TensorToScalars : impl::TensorToScalarsBase<TensorToScalars> {
  using TensorToScalarsBase::TensorToScalarsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    mlir::ConversionTarget target(getContext());

    // do not unroll tensors with more than maxSize elements
    int maxSizeInt = maxSize.getValue();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [maxSizeInt](
            RankedTensorType tensorType,
            SmallVectorImpl<Type>& types) -> std::optional<LogicalResult> {
          if (!tensorType.hasStaticShape()) return std::nullopt;
          if (tensorType.getNumElements() > maxSizeInt) return std::nullopt;
          types = SmallVector<Type>(tensorType.getNumElements(),
                                    tensorType.getElementType());
          return success();
        });
    typeConverter.addSourceMaterialization(buildFromElementsOp);
    typeConverter.addTargetMaterialization(buildExtractOps);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      return typeConverter.isLegal(op.getOperation());
    });
    target.addDynamicallyLegalDialect<func::FuncDialect, scf::SCFDialect>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    RewritePatternSet patterns(context);
    patterns.add<ConvertFromElementsOp, ConvertInsertOp, ConvertExtractOp,
                 ConvertConstantOp>(typeConverter, context);
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                        typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config)))
      signalPassFailure();

    // Empty PatternSet = only run folders (should never fail)
    RewritePatternSet emptyPatterns(context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(emptyPatterns));
  }
};

}  // namespace heir
}  // namespace mlir
