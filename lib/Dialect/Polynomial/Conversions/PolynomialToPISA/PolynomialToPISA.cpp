#include "lib/Dialect/Polynomial/Conversions/PolynomialToPISA/PolynomialToPISA.h"

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/PISA/IR/PISADialect.h"
#include "lib/Dialect/PISA/IR/PISAOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_POLYNOMIALTOPISA
#include "lib/Dialect/Polynomial/Conversions/PolynomialToPISA/PolynomialToPISA.h.inc"

// Remove this class if no type conversions are necessary
class PolynomialToPISATypeConverter : public TypeConverter {
 public:
  PolynomialToPISATypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](polynomial::PolynomialType type) -> Type {
      auto ring = type.getRing();
      auto degree = ring.getPolynomialModulus().getPolynomial().getDegree();
      if (degree != 8192) return nullptr;  // Unsupported -> hard error
      return RankedTensorType::get({degree}, ring.getCoefficientType());
    });
  }
};

struct ConvertAddOp : public OpConversionPattern<polynomial::AddOp> {
  ConvertAddOp(mlir::MLIRContext *context)
      : OpConversionPattern<polynomial::AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      polynomial::AddOp op, polynomial::AddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto polynomialType =
        llvm::cast<polynomial::PolynomialType>(op.getResult().getType());

    auto modArithType = dyn_cast<mod_arith::ModArithType>(
        polynomialType.getRing().getCoefficientType());
    if (!modArithType) {
      op.emitOpError() << "Expected Polynomial's coefficient type to be "
                          "mod_arith type when lowering to PISA.";
      return failure();
    }
    auto q = rewriter.getI32IntegerAttr(modArithType.getModulus().getInt());
    // TODO: add RNS support
    auto i = rewriter.getI32IntegerAttr(0);
    rewriter.replaceOpWithNewOp<pisa::AddOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs(), q, i);
    return success();
  }
};

struct PolynomialToPISA : public impl::PolynomialToPISABase<PolynomialToPISA> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    PolynomialToPISATypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<pisa::PISADialect>();
    target.addIllegalDialect<polynomial::PolynomialDialect>();
    target.addIllegalDialect<mod_arith::ModArithDialect>();

    patterns.add<ConvertAddOp>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    // TODO: Add a pass to split polynomials with degree > 8k into smaller
    // "native" polynomials. This needs to be another OneToN Conversion, as a
    // single polynomial type (with degree >8k) will result in multiple "native"
    // polynomials. For most ops, the translation is trivial (emit affine.for or
    // just emit all ops?) but for NTT/iNTT, it's less trivial
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
