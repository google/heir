#include "lib/Transforms/LowerPolynomialEval/Patterns.h"

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "lower-polynomial-eval"

namespace mlir {
namespace heir {

using polynomial::EvalOp;
using polynomial::TypedFloatPolynomialAttr;

TypedAttr getScalarOrDenseAttr(Type tensorOrScalarType, APFloat value) {
  return TypeSwitch<Type, TypedAttr>(tensorOrScalarType)
      .Case<FloatType>(
          [&](FloatType type) { return FloatAttr::get(type, value); })
      .Case<ShapedType>(
          [&](ShapedType type) { return DenseElementsAttr::get(type, value); })
      .Default([](Type) { return nullptr; });
}

LogicalResult LowerViaHorner::matchAndRewrite(EvalOp op,
                                              PatternRewriter& rewriter) const {
  Type evaluatedType = op.getValue().getType();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  b.setInsertionPoint(op);

  LLVM_DEBUG(llvm::dbgs() << "evaluatedType: " << evaluatedType << "\n");

  auto attr =
      dyn_cast<polynomial::TypedFloatPolynomialAttr>(op.getPolynomialAttr());
  if (!attr) return failure();

  auto terms = attr.getValue().getPolynomial().getTerms();
  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree >= degreeThreshold) return failure();

  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  DenseMap<int64_t, TypedAttr> attributeMap;
  for (auto& [key, monomial] : monomialMap) {
    attributeMap[key] =
        getScalarOrDenseAttr(evaluatedType, monomial.getCoefficient());
  }

  // Start with the coefficient of the highest degree term
  Value result =
      b.create<arith::ConstantOp>(evaluatedType, attributeMap[maxDegree]);

  // Apply Horner's method, accounting for possible missing terms
  auto x = op.getOperand();
  for (int64_t i = maxDegree - 1; i >= 0; i--) {
    // Multiply by x
    result = b.create<arith::MulFOp>(result, x);

    // Add coefficient if this term exists, otherwise continue
    if (attributeMap.find(i) != attributeMap.end()) {
      auto coeffConst =
          b.create<arith::ConstantOp>(evaluatedType, attributeMap[i]);
      result = b.create<arith::AddFOp>(result, coeffConst);
    }
  }

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult LowerViaPatersonStockmeyerMonomial::matchAndRewrite(
    EvalOp op, PatternRewriter& rewriter) const {
  Type evaluatedType = op.getValue().getType();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  b.setInsertionPoint(op);

  auto attr =
      dyn_cast<polynomial::TypedFloatPolynomialAttr>(op.getPolynomialAttr());
  if (!attr) return failure();

  auto terms = attr.getValue().getPolynomial().getTerms();

  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree >= degreeThreshold) return failure();

  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  DenseMap<int64_t, TypedAttr> attributeMap;
  for (auto& [key, monomial] : monomialMap) {
    attributeMap[key] =
        getScalarOrDenseAttr(evaluatedType, monomial.getCoefficient());
  }

  // Choose k optimally - sqrt of maxDegree is typically a good choice
  int64_t k = std::max(static_cast<int64_t>(std::ceil(std::sqrt(maxDegree))),
                       static_cast<int64_t>(1));

  // Precompute x^1, x^2, ..., x^k
  Value x = op.getOperand();
  std::vector<Value> xPowers(k + 1);
  xPowers[0] =
      b.create<arith::ConstantOp>(evaluatedType, b.getOneAttr(evaluatedType));
  xPowers[1] = x;
  for (int64_t i = 2; i <= k; i++) {
    xPowers[i] = b.create<arith::MulFOp>(xPowers[i - 1], x).getResult();
  }

  // Number of chunks we'll need
  int64_t m =
      static_cast<int64_t>(std::ceil(static_cast<double>(maxDegree + 1) / k));
  std::vector<Value> chunkValues(m, nullptr);

  for (int64_t i = 0; i < m; i++) {
    // Start with coefficient of degree (i+1)*k-1, if present
    int64_t highestDegreeInChunk = std::min((i + 1) * k - 1, maxDegree);
    int64_t lowestDegreeInChunk = i * k;

    Value chunkValue = nullptr;
    bool hasTerms = false;

    for (int64_t j = lowestDegreeInChunk; j <= highestDegreeInChunk; j++) {
      if (attributeMap.count(j)) {
        // Get the power index relative to the chunk's starting point
        int64_t powerIndex = j - lowestDegreeInChunk;

        Value coeff =
            b.create<arith::ConstantOp>(evaluatedType, attributeMap[j]);
        Value term;

        if (powerIndex == 0) {
          term = coeff;  // x^0 = 1
        } else {
          term = b.create<arith::MulFOp>(coeff, xPowers[powerIndex]);
        }

        if (!hasTerms) {
          chunkValue = term;
          hasTerms = true;
        } else {
          chunkValue = b.create<arith::AddFOp>(chunkValue, term);
        }
      }
    }

    if (hasTerms) {
      chunkValues[i] = chunkValue;
    } else {
      chunkValues[i] = b.create<arith::ConstantOp>(
          evaluatedType, b.getZeroAttr(evaluatedType));
    }
  }

  // Combine chunks using Horner's method with x^k
  Value result = nullptr;
  bool hasNonEmptyChunk = false;

  for (int64_t i = m - 1; i >= 0; i--) {
    if (chunkValues[i]) {
      if (!hasNonEmptyChunk) {
        // First non-empty chunk encountered
        result = chunkValues[i];
        hasNonEmptyChunk = true;
      } else {
        // Multiply previous result by x^k and add this chunk
        result = b.create<arith::MulFOp>(result, xPowers[k]);
        result = b.create<arith::AddFOp>(result, chunkValues[i]);
      }
    } else if (hasNonEmptyChunk) {
      // Empty chunk but we have previous chunks
      result = b.create<arith::MulFOp>(result, xPowers[k]);
    }
  }

  // Handle the case where no terms were found
  if (!hasNonEmptyChunk) {
    result = b.create<arith::ConstantOp>(evaluatedType,
                                         b.getZeroAttr(evaluatedType));
  }

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult LowerViaPatersonStockmeyerChebyshev::matchAndRewrite(
    EvalOp op, PatternRewriter& rewriter) const {
  return failure();
}

}  // namespace heir
}  // namespace mlir
