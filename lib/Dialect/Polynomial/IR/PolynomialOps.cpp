#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APInt.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

/// A verifier to ensure that a polynomial ring's coefficient type
/// matches the given scalar type. This is useful when verifying an op like
/// mul_scalar, whose arguments must have matching types, but one type
/// is derived from the coefficient type of a polynomial ring attribute.
///
/// polynomialLikeType may be a polynomial or shaped type whose element type is
/// a polynomial type.
///
/// If `op` is provided, any errors will be emitted with the operation's
/// emitOpError.
template <typename Op>
LogicalResult coefficientTypeMatchesScalarType(Type polynomialLikeType,
                                               Type scalarType, Op* op) {
  PolynomialType polyType;

  if (auto shapedPolyType = dyn_cast<ShapedType>(polynomialLikeType)) {
    polyType = cast<PolynomialType>(shapedPolyType.getElementType());
  } else if (isa<PolynomialType>(polynomialLikeType)) {
    polyType = cast<PolynomialType>(polynomialLikeType);
  } else {
    op->emitOpError() << "expected a polynomial or shaped type, found "
                      << polynomialLikeType;
  }

  Type coefficientType = polyType.getRing().getCoefficientType();

  if (coefficientType != scalarType) {
    op->emitOpError() << "polynomial coefficient type " << coefficientType
                      << " does not match scalar type " << scalarType;
    return failure();
  }
  return success();
}

void FromTensorOp::build(OpBuilder& builder, OperationState& result,
                         Value input, RingAttr ring) {
  TensorType tensorType = dyn_cast<TensorType>(input.getType());

  // The input tensor can be a mod_arith type or a plain integer type
  int64_t bitWidth = 0;
  if (auto modArithType =
          dyn_cast<mod_arith::ModArithType>(tensorType.getElementType())) {
    bitWidth = modArithType.getModulus().getType().getIntOrFloatBitWidth();
  } else if (auto intType =
                 dyn_cast<IntegerType>(tensorType.getElementType())) {
    bitWidth = intType.getIntOrFloatBitWidth();
  } else {
    llvm_unreachable("unsupported tensor element type");
  }

  APInt cmod(1 + bitWidth, 1);
  cmod = cmod << bitWidth;
  Type resultType = PolynomialType::get(builder.getContext(), ring);
  build(builder, result, resultType, input);
}

LogicalResult FromTensorOp::verify() {
  ArrayRef<int64_t> tensorShape = getInput().getType().getShape();
  RingAttr ring = getOutput().getType().getRing();
  IntPolynomialAttr polyMod = ring.getPolynomialModulus();

  auto inputEltTy = cast<TensorType>(getInput().getType()).getElementType();
  auto outputPolyLikeTy = getOutput().getType();
  if (failed(coefficientTypeMatchesScalarType(outputPolyLikeTy, inputEltTy,
                                              this))) {
    return failure();
  }

  if (polyMod) {
    unsigned polyDegree = polyMod.getPolynomial().getDegree();
    bool compatible = tensorShape.size() == 1 && tensorShape[0] <= polyDegree;
    if (!compatible) {
      InFlightDiagnostic diag = emitOpError()
                                << "input type " << getInput().getType()
                                << " does not match output type "
                                << getOutput().getType();
      diag.attachNote()
          << "the input type must be a tensor of shape [d] where d "
             "is at most the degree of the polynomialModulus of "
             "the output type's ring attribute";
      return diag;
    }
  }

  return success();
}

LogicalResult ToTensorOp::verify() {
  ArrayRef<int64_t> tensorShape = getOutput().getType().getShape();

  auto outputEltTy = cast<TensorType>(getOutput().getType()).getElementType();
  auto inputPolyLikeTy = getInput().getType();
  if (failed(coefficientTypeMatchesScalarType(inputPolyLikeTy, outputEltTy,
                                              getOperation()))) {
    return emitOpError() << "output tensor element type " << outputEltTy
                         << " does not match input type " << inputPolyLikeTy;
  }

  IntPolynomialAttr polyMod =
      getInput().getType().getRing().getPolynomialModulus();
  if (polyMod) {
    unsigned polyDegree = polyMod.getPolynomial().getDegree();
    bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;

    if (compatible) return success();

    InFlightDiagnostic diag = emitOpError()
                              << "input type " << getInput().getType()
                              << " does not match output type "
                              << getOutput().getType();
    diag.attachNote()
        << "the output type must be a tensor of shape [d] where d "
           "is at most the degree of the polynomialModulus of "
           "the input type's ring attribute";
    return diag;
  }

  return success();
}

LogicalResult ModSwitchOp::verify() {
  if (getInput().getType().getRing().getPolynomialModulus() !=
      getOutput().getType().getRing().getPolynomialModulus()) {
    return emitOpError()
           << "the two polynomials must have the same polynomialModulus";
  }
  return success();
}

/// Test if a value is a primitive nth root of unity modulo cmod.
bool isPrimitiveNthRootOfUnity(const APInt& root, const APInt& n,
                               const APInt& cmod) {
  // The first or subsequent multiplications, may overflow the input bit width,
  // so scale them up to ensure they do not overflow.
  unsigned requiredBitWidth =
      std::max(root.getActiveBits() * 2, cmod.getActiveBits() * 2);
  APInt r = APInt(root).zextOrTrunc(requiredBitWidth);
  APInt cmodExt = APInt(cmod).zextOrTrunc(requiredBitWidth);
  assert(r.ule(cmodExt) && "root must be less than cmod");
  uint64_t upperBound = n.getZExtValue();

  APInt a = r;
  for (size_t k = 1; k < upperBound; k++) {
    if (a.isOne()) return false;
    a = (a * r).urem(cmodExt);
  }
  return a.isOne();
}

/// Verify that the types involved in an NTT or INTT operation are
/// compatible.
static LogicalResult verifyNTTOp(Operation* op, PolynomialType poly,
                                 RankedTensorType tensorType,
                                 std::optional<PrimitiveRootAttr> root) {
  Attribute encoding = tensorType.getEncoding();
  if (!encoding) {
    return op->emitOpError()
           << "expects a ring encoding to be provided to the tensor";
  }
  auto encodedRing = dyn_cast<RingAttr>(encoding);
  if (!encodedRing) {
    return op->emitOpError()
           << "the provided tensor encoding is not a ring attribute";
  }

  RingAttr ring = poly.getRing();
  if (encodedRing != ring) {
    return op->emitOpError()
           << "encoded ring type " << encodedRing
           << " is not equivalent to the polynomial ring " << ring;
  }

  unsigned polyDegree = ring.getPolynomialModulus().getPolynomial().getDegree();
  ArrayRef<int64_t> tensorShape = tensorType.getShape();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;
  if (!compatible) {
    InFlightDiagnostic diag = op->emitOpError()
                              << "tensor type " << tensorType
                              << " does not match output type " << ring;
    diag.attachNote() << "the tensor must have shape [d] where d "
                         "is exactly the degree of the polynomialModulus of "
                         "the polynomial type's ring attribute";
    return diag;
  }

  auto coeffType = dyn_cast<mod_arith::ModArithType>(ring.getCoefficientType());
  if (!coeffType) {
    return op->emitOpError()
           << "expected coefficient type to be mod_arith type";
  }
  if (failed(coefficientTypeMatchesScalarType(poly, tensorType.getElementType(),
                                              op)))
    return failure();

  if (root.has_value()) {
    APInt rootValue = root.value().getValue().getValue();
    APInt rootDegree = root.value().getDegree().getValue();
    auto coeffType =
        dyn_cast<mod_arith::ModArithType>(ring.getCoefficientType());

    if (!coeffType) {
      return op->emitOpError() << "when setting a primitive root, the "
                                  "coefficient type must be mod_arith"
                               << ", but found " << ring.getCoefficientType();
    }

    APInt cmod = coeffType.getModulus().getValue();
    if (!isPrimitiveNthRootOfUnity(rootValue, rootDegree, cmod)) {
      return op->emitOpError()
             << "provided root " << rootValue.getZExtValue()
             << " is not a primitive root " << "of unity mod "
             << cmod.getZExtValue() << ", with the specified degree "
             << rootDegree.getZExtValue();
    }
  }

  return success();
}

LogicalResult NTTOp::verify() {
  return verifyNTTOp(this->getOperation(), getInput().getType(),
                     getOutput().getType(), getRoot());
}

LogicalResult INTTOp::verify() {
  return verifyNTTOp(this->getOperation(), getOutput().getType(),
                     getInput().getType(), getRoot());
}

LogicalResult MulScalarOp::verify() {
  return coefficientTypeMatchesScalarType(getPolynomial().getType(),
                                          getScalar().getType(), this);
}

LogicalResult MonomialOp::verify() {
  return coefficientTypeMatchesScalarType(getOutput().getType(),
                                          getCoefficient().getType(), this);
}

LogicalResult LeadingTermOp::verify() {
  return coefficientTypeMatchesScalarType(getInput().getType(),
                                          getCoefficient().getType(), this);
}

LogicalResult EvalOp::verify() {
  Attribute attr = getPolynomialAttr();
  bool empty =
      TypeSwitch<Attribute, bool>(attr)
          .Case<IntPolynomialAttr>([&](IntPolynomialAttr intAttr) {
            return intAttr.getPolynomial().getTerms().empty();
          })
          .Case<TypedIntPolynomialAttr>([&](TypedIntPolynomialAttr intAttr) {
            return intAttr.getValue().getPolynomial().getTerms().empty();
          })
          .Case<FloatPolynomialAttr>([&](FloatPolynomialAttr floatAttr) {
            return floatAttr.getPolynomial().getTerms().empty();
          })
          .Case<TypedFloatPolynomialAttr>(
              [&](TypedFloatPolynomialAttr floatAttr) {
                return floatAttr.getValue().getPolynomial().getTerms().empty();
              })
          .Default([&](Attribute) { return false; });
  if (empty) {
    return emitError() << "Empty polynomials are not supported for eval op";
  }

  return success();
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  auto loc = parser.getCurrentLocation();

  // Using the built-in parser.parseAttribute requires the full
  // #polynomial.typed_int_polynomial syntax, which is excessive.
  // Instead we parse a keyword int to signal it's an integer polynomial
  Type type;
  if (succeeded(parser.parseOptionalKeyword("float"))) {
    Attribute floatPolyAttr = FloatPolynomialAttr::parse(parser, nullptr);
    if (floatPolyAttr) {
      if (parser.parseColon() || parser.parseType(type)) return failure();
      result.addAttribute("value",
                          TypedFloatPolynomialAttr::get(type, floatPolyAttr));
      result.addTypes(type);
      return success();
    }
  }

  if (succeeded(parser.parseOptionalKeyword("int"))) {
    Attribute intPolyAttr = IntPolynomialAttr::parse(parser, nullptr);
    if (intPolyAttr) {
      if (parser.parseColon() || parser.parseType(type)) return failure();

      result.addAttribute("value",
                          TypedIntPolynomialAttr::get(type, intPolyAttr));
      result.addTypes(type);
      return success();
    }
  }

  // In the worst case, still accept the verbose versions.
  TypedIntPolynomialAttr typedIntPolyAttr;
  OptionalParseResult res =
      parser.parseOptionalAttribute<TypedIntPolynomialAttr>(
          typedIntPolyAttr, "value", result.attributes);
  if (res.has_value() && succeeded(res.value())) {
    result.addTypes(typedIntPolyAttr.getType());
    return success();
  }

  TypedFloatPolynomialAttr typedFloatPolyAttr;
  res = parser.parseAttribute<TypedFloatPolynomialAttr>(
      typedFloatPolyAttr, "value", result.attributes);
  if (res.has_value() && succeeded(res.value())) {
    result.addTypes(typedFloatPolyAttr.getType());
    return success();
  }

  return parser.emitError(
      loc, "Failed to parse polynomimal.constant op for unknown reasons.");
}

void ConstantOp::print(OpAsmPrinter& p) {
  p << " ";
  if (auto intPoly = dyn_cast<TypedIntPolynomialAttr>(getValue())) {
    p << "int";
    intPoly.getValue().print(p);
  } else if (auto floatPoly = dyn_cast<TypedFloatPolynomialAttr>(getValue())) {
    p << "float";
    floatPoly.getValue().print(p);
  } else {
    assert(false && "unexpected attribute type");
  }
  p << " : ";
  p.printType(getOutput().getType());
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext* context, std::optional<mlir::Location> location,
    ConstantOp::Adaptor adaptor,
    llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  Attribute operand = adaptor.getValue();
  if (auto intPoly = dyn_cast<TypedIntPolynomialAttr>(operand)) {
    inferredReturnTypes.push_back(intPoly.getType());
  } else if (auto floatPoly = dyn_cast<TypedFloatPolynomialAttr>(operand)) {
    inferredReturnTypes.push_back(floatPoly.getType());
  } else {
    assert(false && "unexpected attribute type");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "lib/Dialect/Polynomial/IR/PolynomialCanonicalization.cpp.inc"
}  // namespace

void NTTOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<NTTAfterINTT>(context);
}

void INTTOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<INTTAfterNTT>(context);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
