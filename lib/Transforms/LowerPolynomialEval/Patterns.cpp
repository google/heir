#include "lib/Transforms/LowerPolynomialEval/Patterns.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Utils/ArithmeticDag.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
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

static APFloat convertFloatToSemantics(APFloat value,
                                       const llvm::fltSemantics& semantics) {
  if (&value.getSemantics() == &semantics) {
    return value;
  }
  bool losesInfo = false;
  APFloat converted = value;
  converted.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
  return converted;
}

TypedAttr getScalarOrDenseAttr(Type tensorOrScalarType, APFloat value) {
  return TypeSwitch<Type, TypedAttr>(tensorOrScalarType)
      .Case<FloatType>([&](FloatType type) {
        APFloat converted =
            convertFloatToSemantics(value, type.getFloatSemantics());
        return static_cast<TypedAttr>(FloatAttr::get(type, converted));
      })
      .Case<ShapedType>([&](ShapedType type) {
        auto elemType = dyn_cast<FloatType>(type.getElementType());
        if (!elemType) return TypedAttr();
        APFloat converted =
            convertFloatToSemantics(value, elemType.getFloatSemantics());
        return static_cast<TypedAttr>(DenseElementsAttr::get(type, converted));
      })
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
  if (!shouldForce() && maxDegree > degreeThreshold) return failure();

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

class IRMaterializingVisitor : public CachingVisitor<Value, Value> {
 public:
  IRMaterializingVisitor(ImplicitLocOpBuilder& builder, Type evaluatedType)
      : CachingVisitor<Value, Value>(),
        builder(builder),
        evaluatedType(evaluatedType) {}

  Value operator()(const ConstantNode& node) override {
    TypedAttr attr;
    if (isa<FloatType>(evaluatedType)) {
      auto floatTy = cast<FloatType>(evaluatedType);
      APFloat apVal(node.value);
      APFloat converted =
          convertFloatToSemantics(apVal, floatTy.getFloatSemantics());
      attr = static_cast<TypedAttr>(FloatAttr::get(floatTy, converted));
    } else {
      attr =
          static_cast<TypedAttr>(IntegerAttr::get(evaluatedType, node.value));
    }
    return builder.create<arith::ConstantOp>(evaluatedType, attr);
  }

  Value operator()(const LeafNode<Value>& node) override { return node.value; }

  template <typename T, typename FloatOp, typename IntOp>
  Value binop(const T& node) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);
    return TypeSwitch<Type, Value>(evaluatedType)
        .template Case<FloatType>(
            [&](auto ty) { return builder.create<FloatOp>(lhs, rhs); })
        .template Case<IntegerType>(
            [&](auto ty) { return builder.create<IntOp>(lhs, rhs); })
        .Default([&](Type) {
          llvm_unreachable(
              "Unsupported type for binary operation in Chebyshev "
              "evaluation");
          return Value();
        });
  }

  Value operator()(const AddNode<Value>& node) override {
    return binop<AddNode<Value>, arith::AddFOp, arith::AddIOp>(node);
  }

  Value operator()(const SubtractNode<Value>& node) override {
    return binop<SubtractNode<Value>, arith::SubFOp, arith::SubIOp>(node);
  }

  Value operator()(const MultiplyNode<Value>& node) override {
    return binop<MultiplyNode<Value>, arith::MulFOp, arith::MulIOp>(node);
  }

  Value operator()(const PowerNode<Value>& node) override {
    assert(false &&
           "Power operation is not used in Chebyshev Paterson Stockmeyer "
           "algorithm.");
    return Value();
  }

 private:
  ImplicitLocOpBuilder& builder;
  Type evaluatedType;
};

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

  auto xNode = ArithmeticDagNode<Value>::leaf(op.getValue());
  auto resultNode = polynomial::patersonStockmeyerChebyshevPolynomialEvaluation(
      xNode, chebCoeffs);

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = resultNode->visit(visitor);

  rewriter.replaceOp(op, finalOutput);
  return success();
}

}  // namespace heir
}  // namespace mlir
