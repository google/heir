#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h"

#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "llvm/include/llvm/ADT/APFloat.h"               // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ACTIVATIONCANONICALIZATIONS
#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h.inc"

static bool IsOne(mlir::Attribute attr) {
  mlir::FloatType floatTy;
  llvm::APFloat floatVal(0.0);

  if (auto splattr = mlir::dyn_cast_or_null<mlir::SplatElementsAttr>(attr)) {
    floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(splattr.getElementType());
    floatVal = splattr.getValues<llvm::APFloat>()[0];
  } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr)) {
    floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(floatAttr.getType());
    floatVal = floatAttr.getValue();
  }

  if (!floatTy) return false;
  auto one =
      llvm::APFloat::getOne(floatTy.getFloatSemantics(), /*Negative=*/false);
  return floatVal == one;
}

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Transforms/ActivationCanonicalizations/Rewrites.cpp.inc"

// select(a > c, a, c) = max(a, c) for floats. This replaces the DRR
// `SelectGreaterThanEqualFloat` pattern and folds in
// two attr-forwarding behaviors so the polynomial-approximation domain survives
// regardless of where torch-mlir attached it:
//   (a) copy discardable attrs off the select itself (the old DRR behavior),
//   (b) if the domain is still missing, copy `domain_lower`/`domain_upper` from
//       an enclosing `linalg.generic` (a torch ReLU imports as a generic
//       carrying those attrs, with `cmpf+select` in its body) The
//       generic's copy is dropped afterwards by stripForwardedDomains().
// Either way the domain lands on the `arith.maximumf` that
// PolynomialApproximation / ReluViaCompositeSign read; without it they fall
// back to [-1, 1].
struct SelectGreaterThanEqualFloatPattern
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter& rewriter) const override {
    auto cmpOp = op.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmpOp)
      return rewriter.notifyMatchFailure(op, "condition is not arith.cmpf");

    auto pred = cmpOp.getPredicate();
    if (pred != arith::CmpFPredicate::UGT && pred != arith::CmpFPredicate::UGE)
      return rewriter.notifyMatchFailure(op, "predicate is not ugt/uge");

    // Must be the ReLU/max shape: select(a >? c, a, c).
    if (cmpOp.getLhs() != op.getTrueValue() ||
        cmpOp.getRhs() != op.getFalseValue())
      return rewriter.notifyMatchFailure(op,
                                         "operands are not select(a>c,a,c)");

    auto maxOp =
        arith::MaximumFOp::create(rewriter, op.getLoc(), op.getTrueValue(),
                                  op.getFalseValue(), cmpOp.getFastmathAttr());

    // (a) Forward any discardable attrs annotated on the select op itself onto
    // the maximumf (the old DRR `SelectGreaterThanEqualFloat` behavior). Covers
    // IR where the domain is attached directly to the select.
    for (auto attr : op->getDiscardableAttrs())
      maxOp->setAttr(attr.getName(), attr.getValue());

    // (b) If the domain still isn't on the maximumf, it lives on an enclosing
    // linalg.generic instead (where torch-mlir's importer attaches the ReLU
    // domain). Copy it down onto the maximumf — the op PolynomialApproximation
    // actually reads. We only COPY here; the generic's own bounds are dropped
    // afterwards by stripForwardedDomains(). A single generic can hold several
    // ReLUs, so stripping as soon as the first one is rewritten would starve
    // the rest and silently leave them on the default [-1, 1] domain.
    if (auto generic = dyn_cast<linalg::GenericOp>(op->getParentOp())) {
      Attribute lo = generic->getAttr("domain_lower");
      Attribute hi = generic->getAttr("domain_upper");
      if (lo && !maxOp->hasAttr("domain_lower"))
        maxOp->setAttr("domain_lower", lo);
      if (hi && !maxOp->hasAttr("domain_upper"))
        maxOp->setAttr("domain_upper", hi);
    }

    rewriter.replaceOp(op, maxOp.getResult());
    return success();
  }
};

// The domain bounds must end up on exactly one op. Once the patterns above have
// copied an enclosing generic's bounds down onto every ReLU in its body, the
// generic's own copy is redundant, so drop it: leaving the bounds on BOTH makes
// a later activation-lifting pass merge two `domain_lower` entries into one
// dictionary, tripping DictionaryAttr's uniqueness assertion.
static void stripForwardedDomains(Operation* root) {
  root->walk([](linalg::GenericOp generic) {
    if (!generic->hasAttr("domain_lower") && !generic->hasAttr("domain_upper"))
      return;
    // Only strip if the bounds actually made it onto an op inside the body;
    // otherwise there was no ReLU to forward them to and they are still the
    // only record of the domain.
    bool forwarded = false;
    generic->getRegion(0).walk([&](Operation* inner) {
      if (inner->hasAttr("domain_lower") || inner->hasAttr("domain_upper"))
        forwarded = true;
    });
    if (forwarded) {
      generic->removeAttr("domain_lower");
      generic->removeAttr("domain_upper");
    }
  });
}

struct ActivationCanonicalizations
    : impl::ActivationCanonicalizationsBase<ActivationCanonicalizations> {
  using ActivationCanonicalizationsBase::ActivationCanonicalizationsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);
    patterns.add<SelectGreaterThanEqualFloatPattern>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));

    stripForwardedDomains(getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
