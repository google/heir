#include "lib/Conversions/MathToEmitC/MathToEmitC.h"

#include <optional>

#include "mlir/include/mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {
namespace {

void ensureStandardInclude(Operation* op, OpBuilder& builder,
                           StringRef header) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  for (auto include : module.getOps<emitc::IncludeOp>())
    if (include.getIsStandardInclude() && include.getInclude() == header)
      return;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  emitc::IncludeOp::create(builder, op->getLoc(), header,
                           /*isStandardInclude=*/true);
}

struct ConvertSqrt : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      math::SqrtOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    ensureStandardInclude(op, rewriter, "cmath");
    rewriter.replaceOp(op, emitc::CallOpaqueOp::create(
                               rewriter, op.getLoc(), TypeRange{resultType},
                               "std::sqrt", adaptor.getOperand())
                               .getResults());
    return success();
  }
};

struct MathToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  MathToEmitCDialectInterface(Dialect* dialect)
      : ConvertToEmitCPatternInterface(dialect) {}

  void populateConvertToEmitCConversionPatterns(
      ConversionTarget& target, TypeConverter& typeConverter,
      RewritePatternSet& patterns,
      std::optional<bool> /*lowerToCpp*/) const override {
    target.addIllegalDialect<math::MathDialect>();
    patterns.add<ConvertSqrt>(typeConverter, patterns.getContext());
  }
};

}  // namespace

void registerConvertMathToEmitCInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, math::MathDialect* dialect) {
    dialect->addInterfaces<MathToEmitCDialectInterface>();
  });
}

}  // namespace mlir::heir
