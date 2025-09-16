#include "lib/Transforms/LowerPolynomialEval/Patterns.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

#define DEBUG_TYPE "lower-polynomial-eval"

namespace mlir {
namespace heir {

using kernel::ArithmeticDagNode;
using kernel::IRMaterializingVisitor;
using kernel::SSAValue;
using polynomial::EvalOp;
using polynomial::FloatPolynomial;
using polynomial::TypedFloatPolynomialAttr;

LogicalResult LowerViaHorner::matchAndRewrite(EvalOp op,
                                              PatternRewriter& rewriter) const {
  Type evaluatedType = op.getValue().getType();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  b.setInsertionPoint(op);

  LLVM_DEBUG(llvm::dbgs() << "evaluatedType: " << evaluatedType << "\n");

  auto attr =
      dyn_cast<polynomial::TypedFloatPolynomialAttr>(op.getPolynomialAttr());
  if (!attr) return failure();

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();
  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree > degreeThreshold) return failure();

  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  DenseMap<int64_t, TypedAttr> attributeMap;
  for (auto& [key, monomial] : monomialMap) {
    attributeMap.insert(
        {key, getScalarOrDenseAttr(evaluatedType, monomial.getCoefficient())});
  }

  // Start with the coefficient of the highest degree term
  Value result =
      arith::ConstantOp::create(b, evaluatedType, attributeMap[maxDegree]);

  // Apply Horner's method, accounting for possible missing terms
  auto x = op.getOperand();
  for (int64_t i = maxDegree - 1; i >= 0; i--) {
    // Multiply by x
    result = arith::MulFOp::create(b, result, x);

    // Add coefficient if this term exists, otherwise continue
    if (attributeMap.find(i) != attributeMap.end()) {
      auto coeffConst =
          arith::ConstantOp::create(b, evaluatedType, attributeMap.at(i));
      result = arith::AddFOp::create(b, result, coeffConst);
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

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();

  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree > degreeThreshold) return failure();

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
      arith::ConstantOp::create(b, evaluatedType, b.getOneAttr(evaluatedType));
  xPowers[1] = x;
  for (int64_t i = 2; i <= k; i++) {
    if (i % 2 == 0) {
      // x^{2k} = (x^{k})^2
      xPowers[i] =
          arith::MulFOp::create(b, xPowers[i / 2], xPowers[i / 2]).getResult();
    } else {
      // x^{2k+1} = x^{k}x^{k+1}
      xPowers[i] = arith::MulFOp::create(b, xPowers[i / 2], xPowers[i / 2 + 1])
                       .getResult();
    }
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
            arith::ConstantOp::create(b, evaluatedType, attributeMap[j]);
        Value term;

        if (powerIndex == 0) {
          term = coeff;  // x^0 = 1
        } else {
          term = arith::MulFOp::create(b, coeff, xPowers[powerIndex]);
        }

        if (!hasTerms) {
          chunkValue = term;
          hasTerms = true;
        } else {
          chunkValue = arith::AddFOp::create(b, chunkValue, term);
        }
      }
    }

    if (hasTerms) {
      chunkValues[i] = chunkValue;
    } else {
      chunkValues[i] = arith::ConstantOp::create(b, evaluatedType,
                                                 b.getZeroAttr(evaluatedType));
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
        result = arith::MulFOp::create(b, result, xPowers[k]);
        result = arith::AddFOp::create(b, result, chunkValues[i]);
      }
    } else if (hasNonEmptyChunk) {
      // Empty chunk but we have previous chunks
      result = arith::MulFOp::create(b, result, xPowers[k]);
    }
  }

  // Handle the case where no terms were found
  if (!hasNonEmptyChunk) {
    result = arith::ConstantOp::create(b, evaluatedType,
                                       b.getZeroAttr(evaluatedType));
  }

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult LowerViaPatersonStockmeyerChebyshev::matchAndRewrite(
    EvalOp op, PatternRewriter& rewriter) const {
  auto attr = dyn_cast<polynomial::TypedChebyshevPolynomialAttr>(
      op.getPolynomialAttr());
  if (!attr) return failure();
  ArrayAttr chebCoeffsAttr = attr.getValue().getCoefficients();
  SmallVector<double> chebCoeffs =
      llvm::map_to_vector(chebCoeffsAttr, [](Attribute attr) {
        return llvm::cast<FloatAttr>(attr).getValue().convertToDouble();
      });

  // Expect domain attributes to be set.
  auto lowerAttr = op->getAttr("domain_lower");
  auto upperAttr = op->getAttr("domain_upper");
  if (!lowerAttr || !upperAttr) return failure();
  double lower = cast<FloatAttr>(lowerAttr).getValue().convertToDouble();
  double upper = cast<FloatAttr>(upperAttr).getValue().convertToDouble();

  // The Chebyshev polynomial is defined on the interval [-1, 1]. We need to
  // rescale the input x in [lower, upper] to be on this unit interval.
  // The mapping is x -> 2(x-L)/(U-L) - 1 = (2/U-L) * x - (U+L)/(U-L)
  APFloat rescale = APFloat(2 / (upper - lower));
  APFloat shift = APFloat((upper + lower) / (upper - lower));

  auto floatTy = dyn_cast<FloatType>(getElementTypeOrSelf(op));
  if (!floatTy) return failure();

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Value xInput = op.getValue();
  if (!rescale.isExactlyValue(1.0)) {
    xInput =
        arith::MulFOp::create(
            b, xInput,
            arith::ConstantOp::create(
                b, op.getType(), getScalarOrDenseAttr(op.getType(), rescale)))
            .getResult();
  }
  if (!shift.isZero()) {
    xInput =
        arith::AddFOp::create(
            b, xInput,
            arith::ConstantOp::create(
                b, op.getType(), getScalarOrDenseAttr(op.getType(), shift)))
            .getResult();
  }
  SSAValue xNode(xInput);

  auto resultNode = polynomial::patersonStockmeyerChebyshevPolynomialEvaluation(
      xNode, chebCoeffs);

  IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = resultNode->visit(visitor);

  rewriter.replaceOp(op, finalOutput);
  return success();
}

}  // namespace heir
}  // namespace mlir
