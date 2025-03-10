#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define
// ModArithDialect, ModArithTypes, ModArithOps,
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/CommonFolders.h"     // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

#define DEBUG_TYPE "mod-arith"

namespace mlir {
namespace heir {
namespace mod_arith {

void ModArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();
}

/// Ensures that the underlying integer type is wide enough for the coefficient
template <typename OpType>
LogicalResult verifyModArithType(OpType op, ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  unsigned modWidth = modulus.getActiveBits();
  if (modWidth > bitWidth - 1)
    return op.emitOpError()
           << "underlying type's bitwidth must be 1 bit larger than "
           << "the modulus bitwidth, but got " << bitWidth
           << " while modulus requires width " << modWidth << ".";
  return success();
}

template <typename OpType>
LogicalResult verifySameWidth(OpType op, ModArithType modArithType,
                              IntegerType integerType) {
  unsigned bitWidth = modArithType.getModulus().getValue().getBitWidth();
  unsigned intWidth = integerType.getWidth();
  if (intWidth != bitWidth)
    return op.emitOpError()
           << "the result integer type should be of the same width as the "
           << "mod arith type width, but got " << intWidth
           << " while mod arith type width " << bitWidth << ".";
  return success();
}

LogicalResult ExtractOp::verify() {
  auto modArithType = getOperandModArithType(*this);
  auto integerType = getResultIntegerType(*this);
  auto result = verifySameWidth(*this, modArithType, integerType);
  if (result.failed()) return result;
  return verifyModArithType(*this, modArithType);
}

LogicalResult ReduceOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult AddOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult SubOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MulOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MacOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult BarrettReduceOp::verify() {
  auto inputType = getInput().getType();
  unsigned bitWidth;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    bitWidth = tensorType.getElementTypeBitWidth();
  } else {
    auto integerType = dyn_cast<IntegerType>(inputType);
    assert(integerType &&
           "expected input to be a ranked tensor type or integer type");
    bitWidth = integerType.getWidth();
  }
  auto expectedBitWidth = (getModulus() - 1).getActiveBits();
  if (bitWidth < expectedBitWidth || 2 * expectedBitWidth < bitWidth) {
    return emitOpError()
           << "input bitwidth is required to be in the range [w, 2w], where w "
              "is the smallest bit-width that contains the range [0, modulus). "
              "Got "
           << bitWidth << " but w is " << expectedBitWidth << ".";
  }
  if (getModulus().slt(0))
    return emitOpError() << "provided modulus " << getModulus().getSExtValue()
                         << " is not a positive integer.";
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  unsigned minBitwidth = 4;  // bitwidth assigned by parser to integer `1`
  Type parsedType;
  if (parser.parseOptionalKeyword("dense").succeeded()) {
    // Dense case
    // We parse the integers as a list, rather than an ArrayAttr, so we can more
    // easily convert them to the correct bitwidth (ArrayAttr forces I64)
    std::vector<APInt> parsedInts;
    if (parser.parseLess() ||
        parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                       [&] {
                                         APInt parsedInt;
                                         if (parser.parseInteger(parsedInt))
                                           return failure();
                                         parsedInts.push_back(parsedInt);
                                         return success();
                                       }) ||
        parser.parseGreater() || parser.parseColonType(parsedType))
      return failure();
    if (parsedInts.empty())
      return parser.emitError(parser.getNameLoc(),
                              "expected at least one integer in dense list.");

    unsigned maxWidth = 0;
    for (auto &parsedInt : parsedInts) {
      // zero becomes `i64` when parsed, so truncate back down to minBitwidth
      parsedInt = parsedInt.isZero() ? parsedInt.trunc(minBitwidth) : parsedInt;
      maxWidth = std::max(maxWidth, parsedInt.getBitWidth());
    }
    for (auto &parsedInt : parsedInts) {
      parsedInt = parsedInt.zextOrTrunc(maxWidth);
    }
    auto attr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(parsedInts.size())},
                              IntegerType::get(parser.getContext(), maxWidth)),
        parsedInts);
    result.addAttribute("value", attr);
  } else {
    // Scalar case
    APInt parsedInt;
    if (parser.parseInteger(parsedInt) || parser.parseColonType(parsedType))
      return failure();
    // zero becomes `i64` when parsed, so truncate back down to minBitwidth
    if (parsedInt.isZero()) parsedInt = parsedInt.trunc(minBitwidth);
    result.addAttribute(
        "value", IntegerAttr::get(IntegerType::get(parser.getContext(),
                                                   parsedInt.getBitWidth()),
                                  parsedInt));
  }
  result.addTypes(parsedType);
  return success();
}

LogicalResult ConstantOp::verify() {
  auto shapedType = dyn_cast<ShapedType>(getType());
  auto modType = dyn_cast<ModArithType>(getElementTypeOrSelf(getType()));
  auto denseAttr = dyn_cast<DenseIntElementsAttr>(getValue());
  auto intAttr = dyn_cast<IntegerAttr>(getValue());

  assert(modType &&
         "return type should be constrained to "
         "ModArithLike by its ODS definition/type constraints.");

  if (!!shapedType != !!denseAttr)
    return emitOpError("must have shaped type iff value is `dense`.");

  auto modBW = modType.getModulus().getValue().getBitWidth();

  if (intAttr) {
    if (intAttr.getValue().getBitWidth() > modBW)
      return emitOpError(
          "value's bitwidth must not be larger than underlying type.");
    return success();
  }

  if (denseAttr) {
    assert(denseAttr.getShapedType().hasStaticShape() &&
           "dense attribute should be guaranteed to have static shape.");

    if (!shapedType.hasStaticShape() ||
        shapedType.getShape() != denseAttr.getShapedType().getShape())
      return emitOpError("tensor shape must be static and match value shape.");

    if (denseAttr.getElementType().getIntOrFloatBitWidth() > modBW)
      return emitOpError(
          "values's bitwidth must not be larger than underlying type");

    return success();
  }
  // anything else: failure
  return emitOpError("value must be IntegerAttr or DenseIntElementsAttr.");
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getOutput().getType());
}

// constant(c0) -> c0 mod q
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  // Check if the value is an IntegerAttr
  auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getValue());
  if (!intAttr) return {};

  auto modType = dyn_cast_if_present<ModArithType>(getType());
  if (!modType) return {};

  // Retrieve the modulus value and its bit width
  APInt modulus = modType.getModulus().getValue();
  unsigned modBitWidth = modulus.getBitWidth();

  // Extract the actual integer values
  APInt cst = intAttr.getValue();

  // Adjust cst's bit width to match modulus if necessary
  cst = cst.zextOrTrunc(modBitWidth);

  // Fold the constant value
  APInt foldedVal = cst.urem(modulus);

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    llvm::dbgs() << "========================================\n";
    llvm::dbgs() << "  Folding Operation: Constant\n";
    llvm::dbgs() << "----------------------------------------\n";
    llvm::dbgs() << "  Value   : " << cst << "\n";
    llvm::dbgs() << "  Modulus : " << modulus << "\n";
    llvm::dbgs() << "  Folded  : " << foldedVal << "\n";
    llvm::dbgs() << "========================================\n";
  });

  // Create the result
  auto elementType = modType.getModulus().getType();
  return IntegerAttr::get(elementType, foldedVal);
}

/// Helper function to handle common folding logic for binary arithmetic
/// operations.
/// - `opName` is used for debug output.
/// - `foldBinFn` defines how the actual binary operation (+, -, *) should be
/// performed.
template <typename FoldAdaptor, typename FoldBinFn>
static OpFoldResult foldBinModOp(Operation *op, FoldAdaptor adaptor,
                                 FoldBinFn &&foldBinFn,
                                 llvm::StringRef opName) {
  // Check if lhs and rhs are IntegerAttrs
  auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs) return {};

  auto modType = dyn_cast<ModArithType>(op->getResultTypes().front());
  if (!modType) return {};

  // Retrieve the modulus value and its bit width
  APInt modulus = modType.getModulus().getValue();
  unsigned modBitWidth = modulus.getBitWidth();

  // Extract the actual integer values
  APInt lhsVal = lhs.getValue();
  APInt rhsVal = rhs.getValue();

  // Adjust lhsVal and rhsVal bit widths to match modulus if necessary
  lhsVal = lhsVal.zextOrTrunc(modBitWidth);
  rhsVal = rhsVal.zextOrTrunc(modBitWidth);

  // Perform the operation using the provided foldBinFn
  APInt foldedVal = foldBinFn(lhsVal, rhsVal, modulus);

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    llvm::dbgs() << "========================================\n";
    llvm::dbgs() << "  Folding Operation: " << opName << "\n";
    llvm::dbgs() << "----------------------------------------\n";
    llvm::dbgs() << "  LHS     : " << lhsVal << "\n";
    llvm::dbgs() << "  RHS     : " << rhsVal << "\n";
    llvm::dbgs() << "  Modulus : " << modulus << "\n";
    llvm::dbgs() << "  Folded  : " << foldedVal << "\n";
    llvm::dbgs() << "========================================\n";
  });

  // Create the result
  auto elementType = modType.getModulus().getType();
  return IntegerAttr::get(elementType, foldedVal);
}

// add(c0, c1) -> (c0 + c1) mod q
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldBinModOp(
      getOperation(), adaptor,
      [](APInt lhs, APInt rhs, APInt modulus) {
        APInt sum = lhs + rhs;
        return sum.urem(modulus);
      },
      "Add");
}

// sub(c0, c1) -> (c0 - c1) mod q
OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldBinModOp(
      getOperation(), adaptor,
      [](APInt lhs, APInt rhs, APInt modulus) {
        APInt diff = lhs - rhs;
        if (diff.isNegative()) {
          diff += modulus;
        }
        return diff.urem(modulus);
      },
      "Sub");
}

// mul(c0, c1) -> (c0 * c1) mod q
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldBinModOp(
      getOperation(), adaptor,
      [](APInt lhs, APInt rhs, APInt modulus) {
        APInt product = lhs * rhs;
        return product.urem(modulus);
      },
      "Mul");
}

Operation *ModArithDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  auto intAttr = dyn_cast_if_present<IntegerAttr>(value);
  if (!intAttr) return nullptr;
  auto modType = dyn_cast_if_present<ModArithType>(type);
  if (!modType) return nullptr;
  auto op = builder.create<mod_arith::ConstantOp>(loc, modType, intAttr);
  return op.getOperation();
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "lib/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}  // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddZero, AddAddConstant, AddSubConstantRHS, AddSubConstantLHS,
              AddMulNegativeOneRhs, AddMulNegativeOneLhs>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SubZero, SubMulNegativeOneRhs, SubMulNegativeOneLhs,
              SubRHSAddConstant, SubLHSAddConstant, SubRHSSubConstantRHS,
              SubRHSSubConstantLHS, SubLHSSubConstantRHS, SubLHSSubConstantLHS,
              SubSubLHSRHSLHS>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MulZero, MulOne, MulMulConstant>(context);
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
