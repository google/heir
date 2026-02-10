#include "lib/Dialect/CGGI/Conversions/CGGIToSCIFRBool/CGGIToSCIFRBool.h"

#include <cstdint>
#include <ratio>
#include <utility>

#include "lib/Dialect/CGGI/Conversions/CGGIToSCIFRBool/CGGIToSCIFRBool.h.inc"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "cggi-to-scifrbool"

using namespace mlir;
using namespace heir;
using namespace mlir::scifrbool;

namespace mlir {
namespace cornami {

#define GEN_PASS_DEF_CGGITOSCIFRBOOL
#include "lib/Dialect/CGGI/Conversions/CGGIToSCIFRBool/CGGIToSCIFRBool.h.inc"

class CGGIToSCIFRBoolTypeConverter : public TypeConverter {
 public:
  CGGIToSCIFRBoolTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([this](ShapedType type) -> Type {
      return type.cloneWith(type.getShape(),
                            this->convertType(type.getElementType()));
    });
  }
};

/// Convert a func by adding key arguments
struct AddKeyArgs : public OpConversionPattern<func::FuncOp> {
  AddKeyArgs(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op)) {
      return failure();
    }

    Type bskType =
        scifrbool::SCIFRBoolBootstrapKeyStandardType::get(getContext());
    Type kskType = scifrbool::SCIFRBoolKeySwitchKeyType::get(getContext());
    Type serverParamsType =
        scifrbool::SCIFRBoolServerParametersType::get(getContext());

    rewriter.modifyOpInPlace(
        op, [&] { op.insertArgument(0, bskType, nullptr, op.getLoc()); });
    rewriter.modifyOpInPlace(
        op, [&] { op.insertArgument(1, kskType, nullptr, op.getLoc()); });
    rewriter.modifyOpInPlace(op, [&] {
      op.insertArgument(2, serverParamsType, nullptr, op.getLoc());
    });
    return success();
  }
};

template <typename BinOp, typename SCIFRBoolBinOp>
struct ConvertCGGITRBBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    rewriter.replaceOp(
        op, b.create<SCIFRBoolBinOp>(adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ConvertBoolNotOp : public OpConversionPattern<cggi::NotOp> {
  ConvertBoolNotOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::NotOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::NotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    rewriter.replaceOp(op, b.create<scifrbool::NotOp>(adaptor.getInput()));
    return success();
  }
};

struct CGGIToSCIFRBool : impl::CGGIToSCIFRBoolBase<CGGIToSCIFRBool> {
  using CGGIToSCIFRBoolBase::CGGIToSCIFRBoolBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *op = getOperation();

    CGGIToSCIFRBoolTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addLegalDialect<scifrbool::SCIFRBoolDialect>();
    target.addIllegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();

    target.addDynamicallyLegalOp<memref::AllocOp, memref::DeallocOp,
                                 memref::StoreOp, memref::LoadOp,
                                 memref::SubViewOp, memref::CopyOp,
                                 tensor::FromElementsOp, tensor::ExtractOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasKeyArg = op.getFunctionType().getNumInputs() > 0 &&
                       mlir::isa<scifrbool::SCIFRBoolBootstrapKeyStandardType>(
                           *op.getFunctionType().getInputs().begin());
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op) ||
              hasKeyArg);
    });

    target.addLegalOp<mlir::arith::ConstantOp>();

    patterns.add<AddKeyArgs, ConvertCGGITRBBinOp<cggi::AndOp, scifrbool::AndOp>,
                 ConvertCGGITRBBinOp<cggi::NandOp, scifrbool::NandOp>,
                 ConvertCGGITRBBinOp<cggi::OrOp, scifrbool::OrOp>,
                 ConvertCGGITRBBinOp<cggi::NorOp, scifrbool::NorOp>,
                 ConvertCGGITRBBinOp<cggi::XorOp, scifrbool::XorOp>,
                 ConvertCGGITRBBinOp<cggi::XNorOp, scifrbool::XNorOp>,
                 ConvertBoolNotOp, ConvertAny<memref::AllocOp>,
                 ConvertAny<memref::DeallocOp>, ConvertAny<memref::StoreOp>,
                 ConvertAny<memref::LoadOp>, ConvertAny<memref::SubViewOp>,
                 ConvertAny<memref::CopyOp>, ConvertAny<tensor::FromElementsOp>,
                 ConvertAny<tensor::ExtractOp> >(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace cornami
}  // namespace mlir
