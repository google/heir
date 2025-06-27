#include "lib/Transforms/LowerPolynomialEval/Patterns.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Utils/Polynomial/ChebyshevDecomposition.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

#define DEBUG_TYPE "lower-polynomial-eval"

namespace mlir {
namespace heir {

using polynomial::EvalOp;
using polynomial::FloatPolynomial;
using polynomial::TypedFloatPolynomialAttr;

namespace {
std::vector<Value> getPowers(Value x, int64_t k, ImplicitLocOpBuilder& b) {
  std::vector<Value> xPowers(k + 1);
  xPowers[0] =
      b.create<arith::ConstantOp>(x.getType(), b.getOneAttr(x.getType()));
  xPowers[1] = x;
  for (int64_t i = 2; i <= k; i++) {
    if (i % 2 == 0) {
      // x^{2k} = (x^{k})^2
      xPowers[i] =
          b.create<arith::MulFOp>(xPowers[i / 2], xPowers[i / 2]).getResult();
    } else {
      // x^{2k+1} = x^{k}x^{k+1}
      xPowers[i] = b.create<arith::MulFOp>(xPowers[i / 2], xPowers[i / 2 + 1])
                       .getResult();
    }
  }
  return xPowers;
}

bool hasElementsLargerThan(const SmallVector<APFloat>& v, APFloat threshold) {
  for (APFloat a : v) {
    if (llvm::abs(a) >= threshold) return true;
  }
  return false;
}

}  // namespace

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

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();
  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree >= degreeThreshold) return failure();

  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  DenseMap<int64_t, TypedAttr> attributeMap;
  for (auto& [key, monomial] : monomialMap) {
    attributeMap.insert(
        {key, getScalarOrDenseAttr(evaluatedType, monomial.getCoefficient())});
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
          b.create<arith::ConstantOp>(evaluatedType, attributeMap.at(i));
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

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();

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
    if (i % 2 == 0) {
      // x^{2k} = (x^{k})^2
      xPowers[i] =
          b.create<arith::MulFOp>(xPowers[i / 2], xPowers[i / 2]).getResult();
    } else {
      // x^{2k+1} = x^{k}x^{k+1}
      xPowers[i] = b.create<arith::MulFOp>(xPowers[i / 2], xPowers[i / 2 + 1])
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
  // TODO(dvadym): add support for not [-1, 1] range.
  Type evaluatedType = op.getValue().getType();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  b.setInsertionPoint(op);

  auto attr =
      dyn_cast<polynomial::TypedFloatPolynomialAttr>(op.getPolynomialAttr());
  if (!attr) return failure();

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();

  int64_t maxDegree = terms.back().getExponent().getSExtValue();

  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  llvm::SmallVector<APFloat> coeffs(maxDegree + 1, APFloat(0.0));
  for (auto& [key, monomial] : monomialMap) {
    coeffs[key] = monomial.getCoefficient();
  }

  // Choose k optimally - sqrt of maxDegree is typically a good choice
  int64_t k = std::max(static_cast<int64_t>(std::ceil(std::sqrt(maxDegree))),
                       static_cast<int64_t>(1));

  // Precompute T_0(x), T_1(x), ..., T_k(x).
  Value x = op.getOperand();
  std::vector<Value> chebPolynomialValues(k + 1);
  chebPolynomialValues[0] =
      b.create<arith::ConstantOp>(evaluatedType, b.getOneAttr(evaluatedType));
  chebPolynomialValues[1] = x;
  auto number2 = b.create<arith::ConstantOp>(
      evaluatedType, b.getIntegerAttr(evaluatedType, 2));
  for (int64_t i = 2; i <= k; i++) {
    if (i % 2 == 0) {
      // T_{2n}(x) = 2(T_n(x))^2 - 1
      auto tnSquared = b.create<arith::MulFOp>(chebPolynomialValues[i / 2],
                                               chebPolynomialValues[i / 2]);
      auto tnSquaredMul2 = b.create<arith::MulFOp>(tnSquared, number2);
      chebPolynomialValues[i] =
          b.create<arith::SubFOp>(tnSquaredMul2, chebPolynomialValues[0]);
    } else {
      // T_{2n+1}(x) = 2*T_n(x) * T_{n+1}(x) - x
      auto tnMulTnPlusOne = b.create<arith::MulFOp>(
          chebPolynomialValues[i / 2], chebPolynomialValues[i / 2 + 1]);
      auto tnMulTnPlusOneMul2 =
          b.create<arith::MulFOp>(tnMulTnPlusOne, number2);
      chebPolynomialValues[i] = b.create<arith::SubFOp>(tnMulTnPlusOneMul2, x);
    }
  }

  // Precompute (T_k)^0, T_k^1, ..., T_k^l
  std::vector<Value> chebKPolynomialPowers =
      getPowers(chebPolynomialValues.back(), k, b);

  polynomial::ChebyshevDecomposition decomposition =
      polynomial::decompose(coeffs, k);

  Value result = nullptr;
  bool resultEmpty = true;
  const APFloat kMinCoeffs = APFloat(1e-12);
  for (int i = 0; i < decomposition.coeffs.size(); ++i) {
    if (!hasElementsLargerThan(decomposition.coeffs[i], kMinCoeffs)) continue;
    Value pol = nullptr;
    bool polEmpty = true;
    for (int j = 0; j < decomposition.coeffs.size(); ++j) {
      APFloat coeff = decomposition.coeffs[i][j];
      if (llvm::abs(coeff) < kMinCoeffs) continue;
      auto coef = b.create<arith::ConstantOp>(
          evaluatedType,
          b.getFloatAttr(evaluatedType, decomposition.coeffs[i][j]));
      auto term = b.create<arith::MulFOp>(coef, chebPolynomialValues[j]);
      if (polEmpty) {
        pol = term;
        polEmpty = false;
      } else {
        pol = b.create<arith::AddFOp>(pol, term);
      }
    }
    pol = b.create<arith::MulFOp>(pol, chebKPolynomialPowers[i]);
    if (!resultEmpty) {
      result = pol;
      resultEmpty = false;
    } else {
      result = b.create<arith::AddFOp>(result, pol);
    }
  }
  rewriter.replaceOp(op, result);
  return success();
}

}  // namespace heir
}  // namespace mlir
