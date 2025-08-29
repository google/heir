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
#include "lib/Utils/Polynomial/Horner.h"
#include "lib/Utils/Polynomial/PatersonStockmeyer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
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
  auto resultNode = polynomial::hornerMonomialPolynomialEvaluation(
      xNode, coefficients, getMinCoefficientThreshold());

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
      xNode, coefficients, getMinCoefficientThreshold());

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

  auto xNode = ArithmeticDagNode<SSAValue>::leaf(op.getOperand());
  auto resultNode = polynomial::patersonStockmeyerChebyshevPolynomialEvaluation(
      xNode, chebCoeffs, getMinCoefficientThreshold());

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  IRMaterializingVisitor visitor(b, op.getValue().getType());
  Value finalOutput = resultNode->visit(visitor);

  rewriter.replaceOp(op, finalOutput);
  return success();
}

}  // namespace heir
}  // namespace mlir
