#include "lib/Transforms/PolynomialCoeffFolding/PolynomialCoeffFolding.h"

#include <memory>
#include <type_traits>
#include <utility>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_POLYNOMIALCOEFFFOLDING
#include "lib/Transforms/PolynomialCoeffFolding/PolynomialCoeffFolding.h.inc"

namespace {

using polynomial::ChebyshevPolynomial;
using polynomial::EvalOp;
using polynomial::FloatMonomial;
using polynomial::FloatPolynomial;
using polynomial::TypedChebyshevPolynomialAttr;
using polynomial::TypedFloatPolynomialAttr;

// Helper to get a float constant from a value.
bool getFloatConstant(Value value, double& res) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      res = floatAttr.getValue().convertToDouble();
      return true;
    }
  }
  return false;
}

// Pattern 1: Operations Before polynomial.eval
struct FoldOpsBeforeEval : public OpRewritePattern<EvalOp> {
  FoldOpsBeforeEval(MLIRContext* context)
      : OpRewritePattern<EvalOp>(context, 2) {}

  LogicalResult matchAndRewrite(EvalOp op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getValue();
    Operation* defOp = input.getDefiningOp();
    if (!defOp) return failure();

    double c = 0.0;
    Value nonConstOperand;
    bool isAdd = false;
    bool isMul = false;

    if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
      if (getFloatConstant(addOp.getRhs(), c)) {
        nonConstOperand = addOp.getLhs();
      } else if (getFloatConstant(addOp.getLhs(), c)) {
        nonConstOperand = addOp.getRhs();
      } else {
        return failure();
      }
      isAdd = true;
    } else if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
      if (getFloatConstant(mulOp.getRhs(), c)) {
        nonConstOperand = mulOp.getLhs();
      } else if (getFloatConstant(mulOp.getLhs(), c)) {
        nonConstOperand = mulOp.getRhs();
      } else {
        return failure();
      }
      isMul = true;
    } else if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
      if (getFloatConstant(subOp.getRhs(), c)) {
        nonConstOperand = subOp.getLhs();
        c = -c;
        isAdd = true;
      } else {
        return failure();
      }
    } else if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
      if (getFloatConstant(divOp.getRhs(), c)) {
        nonConstOperand = divOp.getLhs();
        c = 1.0 / c;
        isMul = true;
      } else {
        return failure();
      }
    } else {
      return failure();
    }

    Attribute polyAttr = op.getPolynomialAttr();
    FloatPolynomial linearPoly =
        FloatPolynomial::fromCoefficients({c, 1.0});  // default for add: x + c
    if (isMul) {
      linearPoly = FloatPolynomial::fromCoefficients({0.0, c});  // c * x
    }

    if (auto attr = dyn_cast<TypedFloatPolynomialAttr>(polyAttr)) {
      FloatPolynomial poly = attr.getValue().getPolynomial();
      FloatPolynomial newPoly = poly.compose(linearPoly);
      auto newAttr = TypedFloatPolynomialAttr::get(attr.getType(), newPoly);
      rewriter.replaceOpWithNewOp<EvalOp>(op, op.getType(), newAttr,
                                          nonConstOperand);
      return success();
    } else if (auto attr = dyn_cast<TypedChebyshevPolynomialAttr>(polyAttr)) {
      ArrayAttr chebCoeffsAttr = attr.getValue().getCoefficients();
      SmallVector<double> chebCoeffs =
          llvm::map_to_vector(chebCoeffsAttr, [](Attribute a) {
            return llvm::cast<FloatAttr>(a).getValue().convertToDouble();
          });
      ChebyshevPolynomial poly(chebCoeffs);
      ChebyshevPolynomial newPoly = poly.compose(linearPoly);
      auto newAttr = TypedChebyshevPolynomialAttr::get(attr.getType(), newPoly);
      rewriter.replaceOpWithNewOp<EvalOp>(op, op.getType(), newAttr,
                                          nonConstOperand);
      return success();
    }

    return failure();
  }
};

// Pattern 2: Operations After polynomial.eval
template <typename OpTy>
struct FoldOpsAfterEval : public OpRewritePattern<OpTy> {
  FoldOpsAfterEval(MLIRContext* context) : OpRewritePattern<OpTy>(context, 1) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    EvalOp evalOp;
    double c = 0.0;
    bool evalOnLhs = false;

    if (auto eOp = lhs.getDefiningOp<EvalOp>()) {
      evalOp = eOp;
      if (!getFloatConstant(rhs, c)) return failure();
      evalOnLhs = true;
    } else if (auto eOp = rhs.getDefiningOp<EvalOp>()) {
      evalOp = eOp;
      if (!getFloatConstant(lhs, c)) return failure();
    } else {
      return failure();
    }

    bool isAdd = std::is_same<OpTy, arith::AddFOp>::value;
    bool isMul = std::is_same<OpTy, arith::MulFOp>::value;
    bool isSub = std::is_same<OpTy, arith::SubFOp>::value;
    bool isDiv = std::is_same<OpTy, arith::DivFOp>::value;

    if (isSub) {
      if (evalOnLhs) {
        c = -c;
        isAdd = true;
      } else {
        // c - eval(x) -> not supported easily without more complex logic
        return failure();
      }
    } else if (isDiv) {
      if (evalOnLhs) {
        c = 1.0 / c;
        isMul = true;
      } else {
        // c / eval(x) -> not supported
        return failure();
      }
    }

    Attribute polyAttr = evalOp.getPolynomialAttr();

    if (auto attr = dyn_cast<TypedFloatPolynomialAttr>(polyAttr)) {
      FloatPolynomial poly = attr.getValue().getPolynomial();
      if (isAdd) {
        // Add c to constant term
        SmallVector<FloatMonomial> terms = llvm::to_vector(poly.getTerms());
        bool found = false;
        for (auto& term : terms) {
          if (term.getExponent().isZero()) {
            term.setCoefficient(term.getCoefficient() + APFloat(c));
            found = true;
            break;
          }
        }
        if (!found) {
          terms.push_back(FloatMonomial(APFloat(c), 0));
        }
        // Need to re-sort if we added a new term? canonicalize handles it
        // usually if sorted, but let's just rebuild
        auto newPolyOr = FloatPolynomial::fromMonomials(terms);
        if (failed(newPolyOr)) return failure();
        FloatPolynomial newPoly = newPolyOr.value();
        auto newAttr = TypedFloatPolynomialAttr::get(attr.getType(), newPoly);
        rewriter.replaceOpWithNewOp<EvalOp>(op, op.getType(), newAttr,
                                            evalOp.getValue());
        return success();
      } else if (isMul) {
        FloatPolynomial newPoly = poly.scale(APFloat(c));
        auto newAttr = TypedFloatPolynomialAttr::get(attr.getType(), newPoly);
        rewriter.replaceOpWithNewOp<EvalOp>(op, op.getType(), newAttr,
                                            evalOp.getValue());
        return success();
      }
    } else if (auto attr = dyn_cast<TypedChebyshevPolynomialAttr>(polyAttr)) {
      ArrayAttr chebCoeffsAttr = attr.getValue().getCoefficients();
      SmallVector<double> chebCoeffs =
          llvm::map_to_vector(chebCoeffsAttr, [](Attribute a) {
            return llvm::cast<FloatAttr>(a).getValue().convertToDouble();
          });

      if (isAdd) {
        if (chebCoeffs.empty()) {
          chebCoeffs.push_back(c);
        } else {
          chebCoeffs[0] += c;
        }
        ChebyshevPolynomial newPoly(chebCoeffs);
        auto newAttr =
            TypedChebyshevPolynomialAttr::get(attr.getType(), newPoly);
        rewriter.replaceOpWithNewOp<EvalOp>(op, op.getType(), newAttr,
                                            evalOp.getValue());
        return success();
      } else if (isMul) {
        for (auto& coeff : chebCoeffs) {
          coeff *= c;
        }
        ChebyshevPolynomial newPoly(chebCoeffs);
        auto newAttr =
            TypedChebyshevPolynomialAttr::get(attr.getType(), newPoly);
        rewriter.replaceOpWithNewOp<EvalOp>(op, op.getType(), newAttr,
                                            evalOp.getValue());
        return success();
      }
    }

    return failure();
  }
};

struct PolynomialCoeffFolding
    : public impl::PolynomialCoeffFoldingBase<PolynomialCoeffFolding> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldOpsBeforeEval>(context);
    patterns.add<FoldOpsAfterEval<arith::AddFOp>>(context);
    patterns.add<FoldOpsAfterEval<arith::MulFOp>>(context);
    patterns.add<FoldOpsAfterEval<arith::SubFOp>>(context);
    patterns.add<FoldOpsAfterEval<arith::DivFOp>>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createPolynomialCoeffFoldingPass() {
  return std::make_unique<PolynomialCoeffFolding>();
}

}  // namespace heir
}  // namespace mlir
