#include "lib/Dialect/Polynomial/Transforms/SplitRNSNTT.h"

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_SPLITRNSNTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

LogicalResult splitRNSNTTGeneric(Operation* op, PatternRewriter& rewriter,
                                 Value input,
                                 std::optional<PrimitiveRootAttr> primRootAttr,
                                 bool forwardNTT) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  PolynomialType polyTy = dyn_cast<PolynomialType>(input.getType());
  if (!polyTy) {
    return rewriter.notifyMatchFailure(
        op, "SplitRNSNTT doesn't handle tensors of polynomials right now");
  }

  Type coeffType = polyTy.getRing().getCoefficientType();
  auto maCoeffType = dyn_cast<mod_arith::ModArithType>(coeffType);
  if (maCoeffType) {
    // The polynomial is already over ModArith, so nothing to do
    return success();
  }
  auto rnsCoeffType = dyn_cast<rns::RNSType>(coeffType);
  if (!rnsCoeffType) {
    return rewriter.notifyMatchFailure(
        op, "NTT polynomial coefficient is not RNS or ModArith");
  }

  std::optional<rns::RNSAttr> rnsRootAttr;
  if (primRootAttr) {
    rnsRootAttr = dyn_cast<rns::RNSAttr>(primRootAttr->getValue());
    if (!rnsRootAttr) {
      return rewriter.notifyMatchFailure(
          op, "primitive root value is not an RNS attribute");
    }

    if (rnsRootAttr->getType() != rnsCoeffType) {
      return rewriter.notifyMatchFailure(
          op, "Primitive root RNS type does not match polynomial RNS type");
    }
  }

  int numLimbs = rnsCoeffType.getBasisTypes().size();
  for (Type basisType : rnsCoeffType.getBasisTypes()) {
    if (!isa<mod_arith::ModArithType>(basisType)) {
      return rewriter.notifyMatchFailure(
          op, "RNS basis contains a non-ModArith coefficient type");
    }
  }

  SmallVector<Value> transformedLimbs;
  transformedLimbs.reserve(numLimbs);
  for (int i = 0; i < numLimbs; i++) {
    Value modArithComponent =
        ExtractSingleSliceOp::create(b, input, b.getIndexAttr(i));

    std::optional<PrimitiveRootAttr> modArithRootAttr;
    if (rnsRootAttr) {
      modArithRootAttr =
          PrimitiveRootAttr::get(op->getContext(), rnsRootAttr->getValues()[i],
                                 primRootAttr->getDegree());
    }

    Value transformed;
    if (forwardNTT) {
      transformed = modArithRootAttr
                        ? NTTOp::create(b, modArithComponent, *modArithRootAttr)
                        : NTTOp::create(b, modArithComponent);
    } else {
      transformed = modArithRootAttr ? INTTOp::create(b, modArithComponent,
                                                      *modArithRootAttr)
                                     : INTTOp::create(b, modArithComponent);
    }
    transformedLimbs.push_back(transformed);
  }
  Value packedEvalLimbs = PackOp::create(b, transformedLimbs);
  rewriter.replaceOp(op, packedEvalLimbs);
  return success();
}

LogicalResult SplitRNSNTTPattern::matchAndRewrite(
    NTTOp op, PatternRewriter& rewriter) const {
  return splitRNSNTTGeneric(op, rewriter, op.getInput(), op.getRoot(), true);
}

LogicalResult SplitRNSINTTPattern::matchAndRewrite(
    INTTOp op, PatternRewriter& rewriter) const {
  return splitRNSNTTGeneric(op, rewriter, op.getInput(), op.getRoot(), false);
}

struct SplitRNSNTT : impl::SplitRNSNTTBase<SplitRNSNTT> {
  using SplitRNSNTTBase::SplitRNSNTTBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<SplitRNSNTTPattern, SplitRNSINTTPattern>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
