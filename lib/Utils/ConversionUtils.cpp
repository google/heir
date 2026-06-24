#include "lib/Utils/ConversionUtils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Patterns.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Verifier.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

static bool isDynamicOrUnshapedType(Type t) {
  if (!t) return false;
  auto shaped = mlir::dyn_cast<ShapedType>(t);
  return shaped && !shaped.hasStaticShape();
}

static bool hasAnyDynamicOrUnrankedShape(Operation* op,
                                         const TypeConverter* typeConverter) {
  for (Type t : op->getOperandTypes()) {
    if (isDynamicOrUnshapedType(t) ||
        isDynamicOrUnshapedType(typeConverter->convertType(t)))
      return true;
  }
  for (Type t : op->getResultTypes()) {
    if (isDynamicOrUnshapedType(t) ||
        isDynamicOrUnshapedType(typeConverter->convertType(t)))
      return true;
  }
  return false;
}

FailureOr<Operation*> convertAnyOperand(const TypeConverter* typeConverter,
                                        Operation* op, ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) {
  if (typeConverter->isLegal(op)) {
    return failure();
  }

  SmallVector<Type> newOperandTypes;
  SmallVector<Type> newResultTypes;
  auto result =
      typeConverter->convertTypes(op->getResultTypes(), newResultTypes);
  if (failed(result)) return failure();
  if (failed(
          typeConverter->convertTypes(op->getOperandTypes(), newOperandTypes)))
    return failure();

  SmallVector<std::unique_ptr<Region>, 1> regions;
  IRMapping mapping;
  for (auto& r : op->getRegions()) {
    Region* newRegion = new Region(op);
    rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
    if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter)))
      return failure();
    regions.emplace_back(newRegion);
  }

  Operation* newOp = rewriter.create(OperationState(
      op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
      op->getAttrs(), op->getSuccessors(), regions));

  rewriter.replaceOp(op, newOp);
  return newOp;
}

struct ConvertExtract : public OpConversionPattern<tensor::ExtractOp> {
  ConvertExtract(mlir::MLIRContext* context)
      : OpConversionPattern<tensor::ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert a tensor.extract that would type-convert to extracting a tensor to
  // a tensor.extract_slice operation instead. Specifically, this targets
  // extracting SourceType from tensor<...xSourceType>  when SourceType would be
  // type converted to tensor<...>.
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (hasAnyDynamicOrUnrankedShape(op, getTypeConverter())) return failure();

    auto convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    auto resultType = mlir::dyn_cast<RankedTensorType>(convertedType);
    if (!resultType) {
      return failure();
    }
    // replace tensor.extract %t[%i] from tensor<shape x SourceType>
    // with an equivalent tensor.slice from tensor<shape x resultshape>
    auto shape = op.getTensor().getType().getShape();
    auto resultShape = resultType.getShape();

    // expand op's list of indices by appending as many zeros as there are
    // dimension in resultShape
    SmallVector<OpFoldResult> offsets;
    offsets.append(op.getIndices().begin(), op.getIndices().end());
    for (size_t i = 0; i < resultShape.size(); ++i) {
      offsets.push_back(rewriter.getIndexAttr(0));
    }

    // expand resultShape by prepending as many ones as there are dimensions in
    // shape
    SmallVector<OpFoldResult> sizes;
    for (size_t i = 0; i < shape.size(); ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
    }
    for (int64_t i : resultShape) {
      sizes.push_back(rewriter.getIndexAttr(i));
    }

    // strides are all 1, and we need as many as there are dimensions in
    // both shape and resultShape together
    SmallVector<OpFoldResult> strides;
    for (size_t i = 0; i < shape.size() + resultShape.size(); ++i) {
      strides.push_back(rewriter.getIndexAttr(1));
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, adaptor.getTensor(), offsets, sizes, strides);

    return success();
  }
};

struct ConvertInsert : public OpConversionPattern<tensor::InsertOp> {
  ConvertInsert(mlir::MLIRContext* context)
      : OpConversionPattern<tensor::InsertOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert a tensor.insert that would type-convert to inserting a tensor to
  // a tensor.insert_slice operation instead. Specifically, this targets
  // inserting SourceType into tensor<...xSourceType>  when SourceType would be
  // type converted to tensor<...>.
  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (hasAnyDynamicOrUnrankedShape(op, getTypeConverter())) return failure();

    auto convertedType =
        getTypeConverter()->convertType(op.getScalar().getType());
    auto resultType = mlir::dyn_cast<RankedTensorType>(convertedType);
    if (!resultType) {
      return failure();
    }
    // replace tensor.insert %s into %t[%i] with tensor<shape x SourceType>
    // with an equivalent tensor.insert_slice with tensor<shape x resultshape>
    auto shape = op.getDest().getType().getShape();
    auto resultShape = resultType.getShape();

    // expand op's list of indices by appending as many zeros as there are
    // dimension in resultShape
    SmallVector<OpFoldResult> offsets;
    offsets.append(op.getIndices().begin(), op.getIndices().end());
    for (size_t i = 0; i < resultShape.size(); ++i) {
      offsets.push_back(rewriter.getIndexAttr(0));
    }

    // expand resultShape by prepending as many ones as there are dimensions in
    // shape
    SmallVector<OpFoldResult> sizes;
    for (size_t i = 0; i < shape.size(); ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
    }
    for (int64_t i : resultShape) {
      sizes.push_back(rewriter.getIndexAttr(i));
    }

    // strides are all 1, and we need as many as there are dimensions in
    // both shape and resultShape together
    SmallVector<OpFoldResult> strides;
    for (size_t i = 0; i < shape.size() + resultShape.size(); ++i) {
      strides.push_back(rewriter.getIndexAttr(1));
    }

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.getScalar(), adaptor.getDest(), offsets, sizes, strides);

    return success();
  }
};

struct ConvertFromElements
    : public OpConversionPattern<tensor::FromElementsOp> {
  ConvertFromElements(mlir::MLIRContext* context)
      : OpConversionPattern<tensor::FromElementsOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Converts a tensor.from_elements %s0, %s1, ... : tensor<...xSourceType>
  // where SourceType would be type-converted to tensor<...> to
  // a concatenation of the converted operands (with appropriate reshape)
  LogicalResult matchAndRewrite(
      tensor::FromElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (hasAnyDynamicOrUnrankedShape(op, getTypeConverter())) return failure();

    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return failure();
    }
    if (adaptor.getElements().empty()) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, resultType.getShape(),
                                                   resultType.getElementType());
      return success();
    }
    // Expand each of the (converted) operands:
    SmallVector<Value> newOperands;

    for (auto o : adaptor.getElements()) {
      // extend tensor<...xT> to tensor<1x...xT>
      auto tensorType = mlir::dyn_cast<RankedTensorType>(o.getType());
      if (!tensorType) {
        // If the input tensor elements don't lower to a tensor,
        // this pattern doesn't apply.
        return failure();
      }

      auto shape = tensorType.getShape();
      SmallVector<int64_t> newShape(1, 1);
      newShape.append(shape.begin(), shape.end());

      // Create a dense constant for targetShape
      auto shapeOp = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(),
          RankedTensorType::get(newShape.size(), rewriter.getIndexType()),
          rewriter.getIndexTensorAttr(newShape));

      auto reshapeOp = tensor::ReshapeOp::create(
          rewriter, op.getLoc(),
          RankedTensorType::get(newShape, tensorType.getElementType()), o,
          shapeOp);
      newOperands.push_back(reshapeOp);
    }
    // Create the final tensor.concat operation, then restore the original
    // tensor shape with the converted element shape appended.
    auto concatOp =
        tensor::ConcatOp::create(rewriter, op.getLoc(), 0, newOperands);
    Value result = concatOp.getResult();
    // This block fires if, e.g., we are calling `from_elements` on four inputs
    // and producing a 2x2xT tensor instead of a 4xT tensor.
    if (result.getType() != resultType) {
      auto shapeOp = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(),
          RankedTensorType::get(resultType.getRank(), rewriter.getIndexType()),
          rewriter.getIndexTensorAttr(resultType.getShape()));
      result = tensor::ReshapeOp::create(rewriter, op.getLoc(), resultType,
                                         result, shapeOp);
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};

template <typename Op>
struct ConvertOffsetSizeStrideOp : public OpConversionPattern<Op> {
  ConvertOffsetSizeStrideOp(TypeConverter& typeConverter,
                            mlir::MLIRContext* context)
      : OpConversionPattern<Op>(typeConverter, context, /*benefit=*/2) {}

  // Convert a tensor.extract_slice on tensor<...xSourceType> when SourceType
  // type-converts to tensor<...>. The slice metadata is extended with the full
  // converted element shape so the result keeps the original slice dimensions
  // and appends the converted element dimensions.
  LogicalResult matchAndRewrite(
      Op op, OpConversionPattern<Op>::OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (hasAnyDynamicOrUnrankedShape(op, this->getTypeConverter()))
      return failure();

    auto originalResultType =
        dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!originalResultType) return failure();
    auto convertedType =
        this->getTypeConverter()->convertType(originalResultType);
    auto resultType = dyn_cast<RankedTensorType>(convertedType);
    if (!resultType) return failure();

    int64_t extraRank = resultType.getRank() - originalResultType.getRank();
    if (extraRank <= 0) return failure();

    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();
    SmallVector<OpFoldResult> strides = op.getMixedStrides();

    ArrayRef<int64_t> extraShape = resultType.getShape().take_back(extraRank);
    for (int64_t dim : extraShape) {
      if (ShapedType::isDynamic(dim)) return failure();
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(dim));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    Value source = adaptor.getSource();
    if constexpr (std::is_same_v<Op, tensor::ExtractSliceOp>) {
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          op, resultType, source, offsets, sizes, strides);
    } else if constexpr (std::is_same_v<Op, tensor::InsertSliceOp>) {
      rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
          op, source, adaptor.getDest(), offsets, sizes, strides);
    } else {
      return failure();
    }
    return success();
  }
};

struct ConvertReshape : public OpConversionPattern<tensor::ReshapeOp> {
  ConvertReshape(mlir::MLIRContext* context)
      : OpConversionPattern<tensor::ReshapeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (hasAnyDynamicOrUnrankedShape(op, getTypeConverter())) return failure();

    auto convertedType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!convertedType) return failure();
    if (convertedType == op.getResult().getType()) return failure();
    for (int64_t dim : convertedType.getShape()) {
      if (ShapedType::isDynamic(dim)) return failure();
    }

    auto shapeOp = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(convertedType.getRank(), rewriter.getIndexType()),
        rewriter.getIndexTensorAttr(convertedType.getShape()));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, convertedType, adaptor.getSource(), shapeOp);
    return success();
  }
};

struct ConvertSplat : public OpConversionPattern<tensor::SplatOp> {
  ConvertSplat(mlir::MLIRContext* context)
      : OpConversionPattern<tensor::SplatOp>(context, /*benefit=*/2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::SplatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (hasAnyDynamicOrUnrankedShape(op, getTypeConverter())) return failure();

    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!resultType || !inputType) {
      return failure();
    }

    if (resultType.getRank() < inputType.getRank()) {
      return failure();
    }

    auto init =
        tensor::EmptyOp::create(rewriter, op.getLoc(), resultType.getShape(),
                                resultType.getElementType());
    SmallVector<int64_t> dimensions;
    int64_t numOrigDims = resultType.getRank() - inputType.getRank();
    for (int64_t i = 0; i < numOrigDims; i++) {
      dimensions.push_back(i);
    }
    auto broadcast = linalg::BroadcastOp::create(
        rewriter, op.getLoc(), adaptor.getInput(), init, dimensions);
    rewriter.replaceOp(op, broadcast.getResult()[0]);
    return success();
  }
};

void addTensorOfTensorConversionPatterns(TypeConverter& typeConverter,
                                         RewritePatternSet& patterns,
                                         ConversionTarget& target) {
  target.addDynamicallyLegalDialect<tensor::TensorDialect>(
      [&](Operation* op) { return typeConverter.isLegal(op); });

  typeConverter.addConversion([&](TensorType type) -> std::optional<Type> {
    if (isDynamicOrUnshapedType(type)) return std::nullopt;

    if (!typeConverter.isLegal(type.getElementType())) {
      if (auto convertedType =
              typeConverter.convertType(type.getElementType())) {
        if (auto castConvertedType =
                mlir::dyn_cast<RankedTensorType>(convertedType)) {
          if (isDynamicOrUnshapedType(castConvertedType)) return std::nullopt;

          //  Create the combined shape
          auto polyShape = castConvertedType.getShape();
          auto tensorShape = type.getShape();
          SmallVector<int64_t, 4> combinedShape(tensorShape.begin(),
                                                tensorShape.end());
          combinedShape.append(polyShape.begin(), polyShape.end());
          auto combinedType = RankedTensorType::get(
              combinedShape, castConvertedType.getElementType());
          return combinedType;
        }
      }
    }
    return std::nullopt;
  });

  target.addDynamicallyLegalDialect<affine::AffineDialect>(
      [&](Operation* op) { return typeConverter.isLegal(op); });

  patterns.add<ConvertAny<tensor::EmptyOp>, ConvertExtract, ConvertInsert,
               ConvertSplat, ConvertFromElements, ConvertReshape,
               ConvertOffsetSizeStrideOp<tensor::ExtractSliceOp>,
               ConvertOffsetSizeStrideOp<tensor::InsertSliceOp>>(
      typeConverter, patterns.getContext());
}

void addStructuralConversionPatterns(TypeConverter& typeConverter,
                                     RewritePatternSet& patterns,
                                     ConversionTarget& target) {
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

  populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);

  target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineYieldOp>(
      [&](Operation* op) { return typeConverter.isLegal(op); });
  patterns
      .add<ConvertAny<affine::AffineForOp>, ConvertAny<affine::AffineYieldOp>>(
          typeConverter, patterns.getContext());

  target.markUnknownOpDynamicallyLegal([&](Operation* op) {
    // These rules are needed to handle interface ops that are not directly
    // registered as legal/illegal with the target.
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                            typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });
}

void addTensorConversionPatterns(TypeConverter& typeConverter,
                                 RewritePatternSet& patterns,
                                 ConversionTarget& target) {
  patterns.add<ConvertAny<tensor::EmptyOp>, ConvertAny<tensor::InsertOp>,
               ConvertAny<tensor::InsertSliceOp>,
               ConvertAny<tensor::ExtractSliceOp>,
               ConvertAny<tensor::FromElementsOp>,
               ConvertAny<tensor::ExtractOp>, ConvertAny<tensor::SplatOp>>(
      typeConverter, patterns.getContext());

  target.addDynamicallyLegalOp<tensor::EmptyOp, tensor::InsertOp,
                               tensor::InsertSliceOp, tensor::ExtractOp,
                               tensor::ExtractSliceOp, tensor::FromElementsOp,
                               tensor::SplatOp>(
      [&](Operation* op) { return typeConverter.isLegal(op); });
}

void addMemRefConversionPatterns(TypeConverter& typeConverter,
                                 RewritePatternSet& patterns,
                                 ConversionTarget& target) {
  patterns.add<ConvertAny<memref::AllocOp>, ConvertAny<memref::StoreOp>,
               ConvertAny<memref::LoadOp>, ConvertAny<memref::CopyOp>,
               ConvertAny<memref::SubViewOp>, ConvertAny<memref::DeallocOp>,
               ConvertAny<memref::GetGlobalOp>, ConvertAny<memref::GlobalOp>>(
      typeConverter, patterns.getContext());

  target.addDynamicallyLegalOp<memref::AllocOp, memref::StoreOp, memref::LoadOp,
                               memref::CopyOp, memref::SubViewOp,
                               memref::DeallocOp, memref::GetGlobalOp,
                               memref::GlobalOp>(
      [&](Operation* op) { return typeConverter.isLegal(op); });
}

FailureOr<Value> getContextualArgFromFunc(Operation* op, Type argType) {
  for (auto blockArg : op->getParentOfType<func::FuncOp>()
                           .getBody()
                           .getBlocks()
                           .front()
                           .getArguments()) {
    if (blockArg.getType() == argType) {
      return blockArg;
    }
  }
  return failure();
}

}  // namespace heir
}  // namespace mlir
