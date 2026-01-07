#include "lib/Transforms/LowerPolynomialEval/Patterns.h"

#include <cstdint>
#include <map>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/Horner.h"
#include "lib/Utils/Polynomial/PatersonStockmeyer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include <bits/stdc++.h>
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
  auto attr =
      dyn_cast<polynomial::TypedFloatPolynomialAttr>(op.getPolynomialAttr());
  if (!attr) return failure();

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();
  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree > degreeThreshold) return failure();

  // Convert coefficient map to std::map<int64_t, double>
  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  std::map<int64_t, double> coefficients;
  for (auto& [key, monomial] : monomialMap) {
    double coeffValue = monomial.getCoefficient().convertToDouble();
    coefficients[key] = coeffValue;
  }

  // Create ArithmeticDag nodes
  auto xNode =
      kernel::ArithmeticDagNode<kernel::SSAValue>::leaf(op.getOperand());
  auto resultNode =
      polynomial::hornerMonomialPolynomialEvaluation(xNode, coefficients);

  // Use IRMaterializingVisitor to convert to MLIR
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  kernel::IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = resultNode->visit(visitor);

  rewriter.replaceOp(op, finalOutput);
  return success();
}

LogicalResult LowerViaPatersonStockmeyerMonomial::matchAndRewrite(
    EvalOp op, PatternRewriter& rewriter) const {
  auto attr =
      dyn_cast<polynomial::TypedFloatPolynomialAttr>(op.getPolynomialAttr());
  if (!attr) return failure();

  FloatPolynomial polynomial = attr.getValue().getPolynomial();
  auto terms = polynomial.getTerms();
  int64_t maxDegree = terms.back().getExponent().getSExtValue();
  const int degreeThreshold = 5;
  if (!shouldForce() && maxDegree > degreeThreshold) return failure();

  // Convert coefficient map to std::map<int64_t, double>
  auto monomialMap = attr.getValue().getPolynomial().getCoeffMap();
  std::map<int64_t, double> coefficients;
  for (auto& [key, monomial] : monomialMap) {
    double coeffValue = monomial.getCoefficient().convertToDouble();
    coefficients[key] = coeffValue;
  }

  // Create ArithmeticDag nodes
  auto xNode =
      kernel::ArithmeticDagNode<kernel::SSAValue>::leaf(op.getOperand());
  auto resultNode = polynomial::patersonStockmeyerMonomialPolynomialEvaluation(
      xNode, coefficients);

  // Use IRMaterializingVisitor to convert to MLIR
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  kernel::IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = resultNode->visit(visitor);

  rewriter.replaceOp(op, finalOutput);
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

  FloatPolynomial monomialPoly = polynomial::ChebyshevPolynomial(chebCoeffs).toStandardBasis();
  std::map<int64_t, double> monomialCoeffs;
  for (const auto& term : monomialPoly.getTerms()) {
    int64_t degree = term.getExponent().getZExtValue();
    double coeff = term.getCoefficient().convertToDouble();
    monomialCoeffs[degree] = coeff;
  }
  auto chebDag =
    polynomial::patersonStockmeyerChebyshevPolynomialEvaluation(
          xNode, chebCoeffs, getMinCoefficientThreshold());
  auto xDag = kernel::ArithmeticDagNode<kernel::SSAValue>::leaf(xInput);
  auto monoDag =
    polynomial::hornerMonomialPolynomialEvaluation(
        xDag, monomialCoeffs);
  polynomial::test::MultiplicativeDepthVisitorImpl<kernel::SSAValue> depthVisitor;

  double chebDepth = depthVisitor.process(chebDag);
  double monoDepth = depthVisitor.process(monoDag);

  bool useMonomial = chebDepth > monoDepth;

  IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = (useMonomial ? monoDag.get()->visit(visitor) : chebDag.get()->visit(visitor));

  rewriter.replaceOp(op, finalOutput);
  return success();
}

}  // namespace heir
}  // namespace mlir
