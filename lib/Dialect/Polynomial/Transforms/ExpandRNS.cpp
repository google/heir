#include "lib/Dialect/Polynomial/Transforms/ExpandRNS.h"

#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Patterns.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/OneToNTypeConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

using namespace mlir::polynomial;

#define GEN_PASS_DEF_EXPANDRNS
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

static std::optional<Value> buildNToOneCast(OpBuilder &builder,
                                            RNSType resultType,
                                            ValueRange inputs, Location loc) {
  auto cast =
      builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
  return cast->getOpResult(0);
}

static std::optional<SmallVector<Value>> buildOneToNCast(OpBuilder &builder,
                                                         TypeRange resultTypes,
                                                         Value input,
                                                         Location loc) {
  // This pass only operates on RNS types
  RNSType inputType = dyn_cast<RNSType>(input.getType());
  if (!inputType) return {};

  // Create cast ops
  SmallVector<Value> values;
  for (auto t : inputType.getBasisTypes()) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, t, input);
    values.push_back(cast->getOpResult(0));
  }
  return values;
}

template <typename OpTy>
class ConvertBinOp : public OneToNOpConversionPattern<OpTy> {
 public:
  using OneToNOpConversionPattern<OpTy>::OneToNOpConversionPattern;
  using OneToNOpConversionPattern<OpTy>::typeConverter;

  LogicalResult matchAndRewrite(
      OpTy op, typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    // OneToNConversion has no Conversion-level illegality handling
    if (typeConverter->isLegal(op)) return failure();

    // *must* be an RNS Type
    auto rnsType = llvm::cast<RNSType>(op.getType());

    // For each RNS element, create a new op
    SmallVector<Value> results;
    for (size_t i = 0; i < rnsType.getBasisTypes().size(); ++i) {
      auto r = rewriter.create<OpTy>(op.getLoc(), rnsType.getBasisTypes()[i],
                                     adaptor.getLhs()[i], adaptor.getRhs()[i]);
      results.push_back(r);
    }
    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct ExpandRNS : impl::ExpandRNSBase<ExpandRNS> {
  using ExpandRNSBase::ExpandRNSBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    OneToNTypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [](RNSType rnsType,
           SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
          types = SmallVector<Type>(rnsType.getBasisTypes());
          return success();
        });
    typeConverter.addArgumentMaterialization(buildNToOneCast);
    typeConverter.addSourceMaterialization(buildNToOneCast);
    typeConverter.addTargetMaterialization(buildOneToNCast);

    RewritePatternSet patterns(context);
    // TODO: extend to all other polynomial ops!
    patterns.add<ConvertBinOp<AddOp>, ConvertBinOp<SubOp>, ConvertBinOp<MulOp>>(
        typeConverter, context);
    scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
    populateFuncTypeConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialOneToNConversion(
            getOperation(), typeConverter, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
