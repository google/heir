#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APInt.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
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
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

template <typename OpT, typename AttrT, typename PropertyAccessor>
static AttrT getInherentAttrFromAttrsOrProperties(
    DictionaryAttr attrs, OpaqueProperties properties, StringRef attrName,
    PropertyAccessor propertyAccessor) {
  if (attrs) {
    if (auto attr = attrs.get(attrName)) {
      if (auto typedAttr = dyn_cast<AttrT>(attr)) {
        return typedAttr;
      }
      return AttrT();
    }
  }

  if (!properties) {
    return AttrT();
  }

  const auto* opProperties = properties.as<typename OpT::Properties*>();
  if (!opProperties) {
    return AttrT();
  }
  return propertyAccessor(*opProperties);
}

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
static LogicalResult verifyNTTOp(Operation* op, Type inputType, Type outputType,
                                 std::optional<PrimitiveRootAttr> root,
                                 Form expectedInputForm) {
  PolynomialType input;
  PolynomialType output;

  if (auto inputPoly = dyn_cast<PolynomialType>(inputType)) {
    auto outputPoly = dyn_cast<PolynomialType>(outputType);
    if (!outputPoly) {
      return op->emitOpError()
             << "expected output to be a polynomial type, but got "
             << outputType;
    }
    input = inputPoly;
    output = outputPoly;
  } else {
    auto inputShaped = dyn_cast<ShapedType>(inputType);
    auto outputShaped = dyn_cast<ShapedType>(outputType);
    if (!inputShaped || !outputShaped) {
      return op->emitOpError() << "expected input/output to be both polynomial "
                                  "types or both shaped polynomial types, but "
                               << "got " << inputType << " and " << outputType;
    }
    if (inputShaped.getShape() != outputShaped.getShape()) {
      return op->emitOpError()
             << "expected input/output shaped types to have the same shape, "
             << "but got " << inputShaped << " and " << outputShaped;
    }

    auto inputElemPoly = dyn_cast<PolynomialType>(inputShaped.getElementType());
    auto outputElemPoly =
        dyn_cast<PolynomialType>(outputShaped.getElementType());
    if (!inputElemPoly || !outputElemPoly) {
      return op->emitOpError()
             << "expected shaped types with polynomial elements, but got "
             << inputType << " and " << outputType;
    }
    input = inputElemPoly;
    output = outputElemPoly;
  }

  RingAttr inputRing = input.getRing();
  RingAttr outputRing = output.getRing();
  if (outputRing != inputRing) {
    return op->emitOpError()
           << "input ring type " << inputRing
           << " is not equivalent to the output ring " << outputRing;
  }

  Form inputForm = input.getForm();
  Form outputForm = output.getForm();
  if (inputForm != expectedInputForm) {
    return op->emitOpError()
           << "expected input with isCoeffForm=" << expectedInputForm;
  }
  if (inputForm == outputForm) {
    return op->emitOpError() << "input and output form must be different, but "
                                "both have isCoeffForm="
                             << inputForm;
  }

  if (root.has_value()) {
    Attribute rootValue = root.value().getValue();
    APInt rootDegree = root.value().getDegree().getValue();

    LogicalResult coeffCheck =
        TypeSwitch<Type, LogicalResult>(inputRing.getCoefficientType())
            .Case<mod_arith::ModArithType>(
                [&](mod_arith::ModArithType coeffType) -> LogicalResult {
                  auto rootValueType =
                      dyn_cast<mod_arith::ModArithAttr>(rootValue);
                  if (!rootValueType || inputRing.getCoefficientType() !=
                                            rootValueType.getType()) {
                    return op->emitOpError() << "Ring has coefficient type "
                                             << inputRing.getCoefficientType()
                                             << ", but primitive root has type "
                                             << rootValueType.getType();
                  }
                  APInt cmod = coeffType.getModulus().getValue();
                  APInt rootValue = rootValueType.getValue().getValue();
                  if (!isPrimitiveNthRootOfUnity(rootValue, rootDegree, cmod)) {
                    return op->emitOpError()
                           << "provided root " << rootValue.getZExtValue()
                           << " is not a primitive root " << "of unity mod "
                           << cmod.getZExtValue()
                           << ", with the specified degree "
                           << rootDegree.getSExtValue();
                  }
                  return success();
                })
            .Case<rns::RNSType>([&](rns::RNSType coeffType) -> LogicalResult {
              auto rootValueType = dyn_cast<rns::RNSAttr>(rootValue);
              if (!rootValueType ||
                  inputRing.getCoefficientType() != rootValueType.getType()) {
                return op->emitOpError() << "Ring has coefficient type "
                                         << inputRing.getCoefficientType()
                                         << ", but primitive root has type "
                                         << rootValueType.getType();
              }
              auto basis = coeffType.getBasisTypes();
              int rnsLength = basis.size();
              for (int i = 0; i < rnsLength; i++) {
                auto limbType = dyn_cast<mod_arith::ModArithType>(basis[i]);
                APInt cmod = limbType.getModulus().getValue();
                mod_arith::ModArithAttr rootLimbValue =
                    dyn_cast<mod_arith::ModArithAttr>(
                        rootValueType.getValues()[i]);
                if (!rootLimbValue || rootLimbValue.getType() != limbType) {
                  return op->emitOpError()
                         << "Ring has coefficient type "
                         << inputRing.getCoefficientType()
                         << ", but primitive root attr had incorrect limb[" << i
                         << "] = " << rootValueType.getValues()[i];
                }
                APInt rootValue = rootLimbValue.getValue().getValue();
                if (!isPrimitiveNthRootOfUnity(rootValue, rootDegree, cmod)) {
                  return op->emitOpError()
                         << "provided root " << rootValue.getZExtValue()
                         << " is not a primitive root " << "of unity mod "
                         << cmod.getZExtValue()
                         << ", with the specified degree "
                         << rootDegree.getSExtValue();
                }
              }
              return success();
            })
            .Default([&](Type coeffType) -> LogicalResult {
              return op->emitOpError()
                     << "Ring has unsupported coefficient type " << coeffType;
            });
    if (failed(coeffCheck)) return coeffCheck;
  }

  return success();
}

LogicalResult NTTOp::verify() {
  return verifyNTTOp(this->getOperation(), getInput().getType(),
                     getOutput().getType(), getRoot(), Form::COEFF);
}

LogicalResult INTTOp::verify() {
  return verifyNTTOp(this->getOperation(), getInput().getType(),
                     getOutput().getType(), getRoot(), Form::EVAL);
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

LogicalResult ExtractSliceOp::verify() {
  auto polyType = dyn_cast<PolynomialType>(getInput().getType());
  if (!polyType) {
    return failure();
  }
  auto rnsType =
      dyn_cast<rns::RNSType>(polyType.getRing().getCoefficientType());
  if (!rnsType) {
    return failure();
  }
  int64_t start = getStart().getZExtValue();
  int64_t size = getSize().getZExtValue();
  return verifyExtractSliceOp(this, rnsType, start, size);
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

static LogicalResult inferNTTReturnType(MLIRContext* ctx, Type inputType,
                                        SmallVectorImpl<Type>& results) {
  auto flipForm = [](Form form) {
    return form == Form::COEFF ? Form::EVAL : Form::COEFF;
  };

  PolynomialType inputPolyTy = dyn_cast<PolynomialType>(inputType);
  RankedTensorType tensorTy = dyn_cast<RankedTensorType>(inputType);
  if (!inputPolyTy) {
    if (!tensorTy) {
      return failure();
    }
    inputPolyTy = dyn_cast<PolynomialType>(tensorTy.getElementType());
    if (!inputPolyTy) {
      return failure();
    }
  }
  PolynomialType outputPolyTy = PolynomialType::get(
      ctx, inputPolyTy.getRing(), flipForm(inputPolyTy.getForm()));
  if (dyn_cast<PolynomialType>(inputType)) {
    results.push_back(outputPolyTy);
  } else {
    results.push_back(RankedTensorType::get(tensorTy.getShape(), outputPolyTy,
                                            tensorTy.getEncoding()));
  }
  return success();
}

LogicalResult NTTOp::inferReturnTypes(MLIRContext* ctx, std::optional<Location>,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type>& results) {
  if (operands.empty()) return failure();
  return inferNTTReturnType(ctx, operands.front().getType(), results);
}

LogicalResult INTTOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  if (operands.empty()) return failure();
  return inferNTTReturnType(ctx, operands.front().getType(), results);
}

LogicalResult ConvertBasisOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange /*regions*/, SmallVectorImpl<Type>& results) {
  if (operands.empty()) return failure();
  Type inputType = operands[0].getType();
  PolynomialType inputPolyType = dyn_cast<PolynomialType>(inputType);
  RankedTensorType inputTensorType;
  if (!inputPolyType) {
    inputTensorType = dyn_cast<RankedTensorType>(inputType);
    if (!inputTensorType) return failure();
    inputPolyType = dyn_cast<PolynomialType>(inputTensorType.getElementType());
    if (!inputPolyType) return failure();
  }
  polynomial::RingAttr ringAttr = inputPolyType.getRing();

  TypeAttr targetBasisAttr =
      getInherentAttrFromAttrsOrProperties<ConvertBasisOp, TypeAttr>(
          attrs, properties, "targetBasis",
          [](const ConvertBasisOp::Properties& prop) {
            return prop.targetBasis;
          });
  if (!targetBasisAttr) {
    return failure();
  }

  rns::RNSType elementType = dyn_cast<rns::RNSType>(targetBasisAttr.getValue());
  if (!elementType) {
    return failure();
  }
  polynomial::RingAttr outputRingAttr = polynomial::RingAttr::get(
      ctx, elementType, ringAttr.getPolynomialModulus());
  PolynomialType resultType = PolynomialType::get(ctx, outputRingAttr);
  if (dyn_cast<PolynomialType>(inputType)) {
    results.push_back(resultType);
  } else {
    results.push_back(RankedTensorType::get(
        inputTensorType.getShape(), resultType, inputTensorType.getEncoding()));
  }
  return success();
}

LogicalResult ExtractSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange /*regions*/, SmallVectorImpl<Type>& results) {
  if (operands.empty()) return failure();
  auto polyType = dyn_cast<PolynomialType>(operands[0].getType());
  if (!polyType) return failure();
  RingAttr ringAttr = polyType.getRing();
  rns::RNSType elementType =
      dyn_cast<rns::RNSType>(ringAttr.getCoefficientType());
  if (!elementType) return failure();

  IntegerAttr startAttr =
      getInherentAttrFromAttrsOrProperties<ExtractSliceOp, IntegerAttr>(
          attrs, properties, "start",
          [](const ExtractSliceOp::Properties& prop) { return prop.start; });
  IntegerAttr sizeAttr =
      getInherentAttrFromAttrsOrProperties<ExtractSliceOp, IntegerAttr>(
          attrs, properties, "size",
          [](const ExtractSliceOp::Properties& prop) { return prop.size; });
  if (!startAttr || !sizeAttr) return failure();

  struct ExtractSliceInferenceAdaptor {
    IntegerAttr start;
    IntegerAttr size;
    APInt getStart() const { return start.getValue(); }
    APInt getSize() const { return size.getValue(); }
  };
  ExtractSliceInferenceAdaptor op{startAttr, sizeAttr};
  rns::RNSType outputRNSType =
      inferExtractSliceReturnTypes(context, &op, elementType);
  RingAttr outputRingAttr =
      RingAttr::get(outputRNSType, ringAttr.getPolynomialModulus());
  results.push_back(PolynomialType::get(context, outputRingAttr));
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
