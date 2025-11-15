#include "lib/Transforms/LowerPolynomialEval/Patterns.h"

#include <algorithm>
#include <cmath>
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

#define DEBUG_TYPE "lower-polynomial-eval"

namespace mlir {
namespace heir {

using kernel::ArithmeticDagNode;
using kernel::IRMaterializingVisitor;
using kernel::SSAValue;
using polynomial::EvalOp;
using polynomial::FloatPolynomial;
using polynomial::TypedFloatPolynomialAttr;

namespace {

// Estimate the number of arithmetic operations for a monomial polynomial
// evaluation using Horner's method.
int64_t estimateHornerOperations(const std::map<int64_t, double>& coeffs) {
  if (coeffs.empty()) return 0;
  int64_t maxDegree = coeffs.rbegin()->first;
  // Horner's method: maxDegree multiplications + (number of non-zero coeffs - 1) additions
  return maxDegree + (coeffs.size() - 1);
}

// Estimate the number of arithmetic operations for Chebyshev Paterson-Stockmeyer.
// This is a rough estimate based on the baby-step giant-step approach.
int64_t estimateChebyshevPSOperations(int64_t degree) {
  if (degree <= 0) return 0;
  
  // Choose k optimally
  int64_t k = std::max(static_cast<int64_t>(std::ceil(std::sqrt(degree))),
                       static_cast<int64_t>(1));
  int64_t l = (degree + k - 1) / k;  // Number of blocks
  
  // Computing T_0 to T_k requires approximately k multiplications (baby step)
  // Computing powers of T_k up to l requires approximately log2(l) multiplications (giant step)
  // Final evaluation requires approximately l * k operations
  int64_t babyStepOps = k;
  int64_t giantStepOps = std::max(static_cast<int64_t>(std::ceil(std::log2(l + 1))), static_cast<int64_t>(0));
  int64_t evaluationOps = l * k;
  
  return babyStepOps + giantStepOps + evaluationOps;
}

// Check if the monomial basis has numerical stability issues.
// Returns true if coefficients are well-conditioned.
bool isMonomialBasisStable(const std::map<int64_t, double>& coeffs,
                          double stabilityThreshold = 1e-10) {
  if (coeffs.empty()) return true;
  
  // Find the maximum absolute value of coefficients
  double maxCoeff = 0.0;
  for (const auto& [degree, coeff] : coeffs) {
    maxCoeff = std::max(maxCoeff, std::abs(coeff));
  }
  
  if (maxCoeff == 0.0) return true;
  
  // Check if any high-degree coefficients are very small relative to max
  int64_t maxDegree = coeffs.rbegin()->first;
  for (const auto& [degree, coeff] : coeffs) {
    // High-degree terms with very small coefficients can cause instability
    if (degree > maxDegree / 2 && std::abs(coeff) < stabilityThreshold * maxCoeff) {
      return false;
    }
  }
  
  return true;
}

// Decide whether to use monomial basis instead of Chebyshev PS.
// Returns true if monomial basis should be preferred.
bool shouldUseMonomialBasis(const SmallVector<double>& chebCoeffs,
                           const std::map<int64_t, double>& monomialCoeffs,
                           double minCoeffThreshold) {
  // Filter out very small coefficients from both representations
  std::map<int64_t, double> filteredMonomial;
  for (const auto& [deg, coeff] : monomialCoeffs) {
    if (std::abs(coeff) >= minCoeffThreshold) {
      filteredMonomial[deg] = coeff;
    }
  }
  
  // If the monomial basis is empty after filtering, don't use it
  if (filteredMonomial.empty()) return false;
  
  // Check numerical stability
  if (!isMonomialBasisStable(filteredMonomial)) {
    return false;
  }
  
  // Estimate operation counts
  int64_t chebDegree = chebCoeffs.size() - 1;
  int64_t monomialOps = estimateHornerOperations(filteredMonomial);
  int64_t chebyshevOps = estimateChebyshevPSOperations(chebDegree);
  
  LLVM_DEBUG({
    llvm::dbgs() << "Chebyshev degree: " << chebDegree << "\n";
    llvm::dbgs() << "Monomial operations estimate: " << monomialOps << "\n";
    llvm::dbgs() << "Chebyshev PS operations estimate: " << chebyshevOps << "\n";
  });
  
  // Use monomial basis if it requires significantly fewer operations
  // Use a threshold to account for estimation inaccuracies
  return monomialOps < chebyshevOps * 0.75;
}

}  // namespace

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

  // Convert Chebyshev polynomial to monomial basis to check if it's simpler
  SmallVector<APFloat> chebCoeffsAPFloat;
  for (double coeff : chebCoeffs) {
    chebCoeffsAPFloat.push_back(APFloat(coeff));
  }
  polynomial::ChebyshevPolynomial chebPoly(chebCoeffsAPFloat);
  FloatPolynomial monomialPoly = chebPoly.toStandardBasis();
  
  // Extract monomial coefficients
  std::map<int64_t, double> monomialCoeffs;
  for (const auto& term : monomialPoly.getTerms()) {
    int64_t degree = term.getExponent().getZExtValue();
    double coeff = term.getCoefficient().convertToDouble();
    monomialCoeffs[degree] = coeff;
  }
  
  // Decide whether to use monomial basis
  if (shouldUseMonomialBasis(chebCoeffs, monomialCoeffs, getMinCoefficientThreshold())) {
    LLVM_DEBUG(llvm::dbgs() << "Using monomial basis for Chebyshev polynomial\n");
    
    // Use Horner's method for the monomial polynomial
    auto xNode = kernel::ArithmeticDagNode<kernel::SSAValue>::leaf(op.getOperand());
    auto resultNode = polynomial::hornerMonomialPolynomialEvaluation(xNode, monomialCoeffs);
    
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    kernel::IRMaterializingVisitor visitor(b, op.getValue().getType());
    Value finalOutput = resultNode->visit(visitor);
    
    rewriter.replaceOp(op, finalOutput);
    return success();
  }

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
      xNode, chebCoeffs, getMinCoefficientThreshold());

  IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = resultNode->visit(visitor);

  rewriter.replaceOp(op, finalOutput);
  return success();
}

}  // namespace heir
}  // namespace mlir
