#include "lib/Conversion/Utils.h"

#include <cstdint>

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Patterns.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm - project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

struct ConvertAny : public ConversionPattern {
  ConvertAny(const TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newOperandTypes;
    if (failed(getTypeConverter()->convertTypes(op->getOperandTypes(),
                                                newOperandTypes)))
      return failure();

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();

    SmallVector<std::unique_ptr<Region>, 1> regions;
    IRMapping mapping;
    for (auto &r : op->getRegions()) {
      Region *newRegion = new Region();
      rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
      if (failed(rewriter.convertRegionTypes(newRegion, *this->typeConverter)))
        return failure();
      regions.emplace_back(newRegion);
    }

    Operation *newOp = rewriter.create(OperationState(
        op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
        op->getAttrs(), op->getSuccessors(), regions));

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<tensor::ExtractOp> {
  ConvertExtract(mlir::MLIRContext *context)
      : OpConversionPattern<tensor::ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert a tensor.extract that would type-convert to extracting a tensor to
  // a tensor.extract_slice operation instead. Specifically, this targets
  // extracting SourceType from tensor<...xSourceType>  when SourceType would be
  // type converted to tensor<...>.
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // replace tensor.extract %t[%i] from tensor<shape x SourceType>
    // with an equivalent tensor.slice from tensor<shape x resultshape>
    auto shape = op.getTensor().getType().getShape();
    auto resultType = getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .cast<RankedTensorType>();
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
    for (long i : resultShape) {
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
  ConvertInsert(mlir::MLIRContext *context)
      : OpConversionPattern<tensor::InsertOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert a tensor.insert that would type-convert to inserting a tensor to
  // a tensor.insert_slice operation instead. Specifically, this targets
  // inserting SourceType into tensor<...xSourceType>  when SourceType would be
  // type converted to tensor<...>.
  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // replace tensor.insert %s into %t[%i] with tensor<shape x SourceType>
    // with an equivalent tensor.insert_slice with tensor<shape x resultshape>
    auto shape = op.getDest().getType().getShape();
    auto resultType = getTypeConverter()
                          ->convertType(op.getScalar().getType())
                          .cast<RankedTensorType>();
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
    for (long i : resultShape) {
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
  ConvertFromElements(mlir::MLIRContext *context)
      : OpConversionPattern<tensor::FromElementsOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Converts a tensor.from_elements %s0, %s1, ... : tensor<...xSourceType>
  // where SourceType would be type-converted to tensor<...> to
  // a concatenation of the converted operands (with appropriate reshape)
  LogicalResult matchAndRewrite(
      tensor::FromElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Expand each of the (converted) operands:
    SmallVector<Value> newOperands;
    for (auto o : adaptor.getElements()) {
      // extend tensor<...xT> to tensor<1x...xT>
      if (auto tensorType = o.getType().dyn_cast<RankedTensorType>()) {
        auto shape = tensorType.getShape();
        SmallVector<int64_t> newShape(1, 1);
        newShape.append(shape.begin(), shape.end());

        // Create a dense constant for targetShape
        auto shapeOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(),
            RankedTensorType::get(newShape.size(), rewriter.getIndexType()),
            rewriter.getIndexTensorAttr(newShape));

        auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
            op.getLoc(),
            RankedTensorType::get(newShape, tensorType.getElementType()), o,
            shapeOp);
        newOperands.push_back(reshapeOp);
      } else {
        newOperands.push_back(o);
      }
    }
    // Create the final tensor.concat operation
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, 0, newOperands);

    return success();
  }
};

void addTensorOfTensorConversionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target) {
  typeConverter.addConversion([&](TensorType type) -> Type {
    if (!typeConverter.isLegal(type.getElementType())) {
      typeConverter.convertType(type.getElementType()).dump();
      if (auto convertedType =
              typeConverter.convertType(type.getElementType())) {
        if (auto castConvertedType =
                convertedType.dyn_cast<RankedTensorType>()) {
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
    return type;
  });

  target.addDynamicallyLegalDialect<tensor::TensorDialect>([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  target.addDynamicallyLegalDialect<affine::AffineDialect>([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  patterns.add<ConvertAny, ConvertExtract, ConvertInsert, ConvertFromElements>(
      typeConverter, patterns.getContext());
}

void addStructuralConversionPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target) {
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
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                            typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });

  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
}

}  // namespace heir
}  // namespace mlir
