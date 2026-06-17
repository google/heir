#include "lib/Dialect/ModArith/IR/ModArithOps.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
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
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
// NOLINTEND(misc-include-cleaner)

#define DEBUG_TYPE "mod-arith"

namespace mlir {
namespace heir {
namespace mod_arith {

template <typename OpType>
LogicalResult verifySameWidth(OpType op, ModQTypeInterface modularType,
                              IntegerType integerType) {
  auto loweringType = modularType.getLoweringType();
  auto storageType = dyn_cast<IntegerType>(getElementTypeOrSelf(loweringType));
  if (!storageType) {
    return op.emitOpError()
           << "expected modular type lowering type to have integer element "
              "type";
  }

  unsigned bitWidth = storageType.getWidth();
  unsigned intWidth = integerType.getWidth();
  if (intWidth != bitWidth)
    return op.emitOpError()
           << "the result integer type should be of the same width as the "
           << "mod arith type width, but got " << intWidth
           << " while mod arith type width " << bitWidth << ".";
  return success();
}

std::vector<int64_t> getShapeOrEmpty(Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    return tensorType.getShape();
  }
  return {};
}

template <typename OpType>
bool isShapeCorrect(OpType op, int64_t rnsLength,
                    std::vector<int64_t>& modularShape,
                    std::vector<int64_t>& integerShape) {
  auto tmp = modularShape;
  if (rnsLength != 0) tmp.push_back(rnsLength);
  return tmp == integerShape;
}

std::optional<ModQTypeInterface> getModQTypeInterface(Type type) {
  if (auto iface = dyn_cast<ModQTypeInterface>(getElementTypeOrSelf(type))) {
    return iface;
  }
  return std::nullopt;
}

std::optional<ModArithType> getResidueModArithType(ModQTypeInterface type,
                                                   unsigned index) {
  if (index >= type.getNumResidues()) {
    return std::nullopt;
  }
  if (auto residueType = dyn_cast<ModArithType>(type.getResidueType(index))) {
    return residueType;
  }
  return std::nullopt;
}

template <typename OpType>
LogicalResult verifyTypeLowering(OpType op, Type modularType,
                                 Type integerType) {
  auto modQType = getModQTypeInterface(modularType);
  if (!modQType) {
    return op.emitOpError()
           << "expected modular type to implement " << "ModQTypeInterface";
  }

  int64_t rnsLength = isa<ShapedType>(modQType->getLoweringType())
                          ? modQType->getNumResidues()
                          : 0;
  auto innerIntegerType = cast<IntegerType>(getElementTypeOrSelf(integerType));
  auto modularShape = getShapeOrEmpty(modularType);
  auto integerShape = getShapeOrEmpty(integerType);
  if (!isShapeCorrect(op, rnsLength, modularShape, integerShape)) {
    return op.emitOpError() << "The shape of input/output type is not correct.";
  }
  return verifySameWidth(op, *modQType, innerIntegerType);
}

LogicalResult EncapsulateOp::verify() {
  return verifyTypeLowering(*this, getOutput().getType(), getInput().getType());
}

LogicalResult LiftOp::verify() {
  return verifyTypeLowering(*this, getInput().getType(), getOutput().getType());
}

template <typename OpType>
LogicalResult verifySingleToMultiModSwitch(OpType op,
                                           ModQTypeInterface singleTy,
                                           ModQTypeInterface multiTy,
                                           bool allowInjection = false) {
  auto singleModArithType = getResidueModArithType(singleTy, 0);
  if (!singleModArithType) {
    return op.emitOpError()
           << "expected single-residue modular type to have a ModArith residue";
  }

  if (multiTy.getNumResidues() < 2) {
    return op.emitOpError() << "expected multi-residue modular type";
  }

  auto bigModulus = (*singleModArithType).getModulus().getValue();
  auto width = bigModulus.getBitWidth();
  auto firstResidueType = getResidueModArithType(multiTy, 0);
  if (!firstResidueType) {
    return op.emitOpError()
           << "expected multi-residue modular type to have ModArith residues";
  }

  auto rnsWidth = (*firstResidueType).getModulus().getValue().getBitWidth();
  if (width < rnsWidth) {
    return op.emitOpError() << "The input and output type can't both be RNS";
  }

  APInt product(width, 1);
  for (unsigned i = 0; i < multiTy.getNumResidues(); ++i) {
    auto residueType = getResidueModArithType(multiTy, i);
    if (!residueType) {
      return op.emitOpError()
             << "expected multi-residue modular type to have ModArith residues";
    }
    auto modulus = (*residueType).getModulus().getValue();
    product *= modulus.zext(width);
  }

  if (product != bigModulus) {
    if (allowInjection && bigModulus.slt(product)) {
      return success();
    }
    return op.emitOpError() << "The product of RNS modulus should equal to the "
                               "modulus of ModArithType";
  }
  return success();
}

LogicalResult ModSwitchOp::verify() {
  auto inputType = getModQTypeInterface(getInput().getType());
  auto outputType = getModQTypeInterface(getOutput().getType());

  if (!inputType || !outputType) {
    llvm_unreachable("Verifier should make sure this doesn't happen.");
  }

  unsigned inputResidues = inputType->getNumResidues();
  unsigned outputResidues = outputType->getNumResidues();
  if (inputResidues == 1 && outputResidues == 1) {
    return success();
  }
  if (inputResidues == 1 && outputResidues > 1) {
    return verifySingleToMultiModSwitch(*this, *inputType, *outputType,
                                        /*allowInjection=*/true);
  }
  if (inputResidues > 1 && outputResidues == 1) {
    return verifySingleToMultiModSwitch(*this, *outputType, *inputType);
  }
  if (inputResidues > 1 && outputResidues > 1) {
    return emitOpError() << "The input and output type can't both be RNS";
  }

  llvm_unreachable("Verifier should make sure this doesn't happen.");
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
    return emitOpError() << "input bitwidth is required to be in the range "
                            "[w, 2w], where w "
                            "is the smallest bit-width that contains the "
                            "range [0, modulus). "
                            "Got "
                         << bitWidth << " but w is " << expectedBitWidth << ".";
  }
  if (getModulus().slt(0))
    return emitOpError() << "provided modulus " << getModulus().getSExtValue()
                         << " is not a positive integer.";
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  unsigned minBitwidth = 4;  // bitwidth assigned by parser to integer `1`
  Type parsedType;
  if (parser.parseOptionalKeyword("dense").succeeded()) {
    // Dense case
    // We parse the integers as a list, rather than an ArrayAttr, so we can
    // more easily convert them to the correct bitwidth (ArrayAttr forces I64)
    std::vector<APInt> parsedInts;
    if (parser.parseLess() ||
        parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Square,
            [&] {
              APInt parsedInt;
              if (parser.parseInteger(parsedInt))
                return parser.emitError(
                           parser.getNameLoc(),
                           "failed to parse integer in dense list"),
                       failure();
              parsedInts.push_back(parsedInt);
              return success();
            }) ||
        parser.parseGreater() || parser.parseColonType(parsedType))
      return parser.emitError(parser.getNameLoc(),
                              "failed to parse dense constant"),
             failure();
    if (parsedInts.empty())
      return parser.emitError(parser.getNameLoc(),
                              "expected at least one integer in dense list.");

    unsigned maxWidth = 0;
    for (auto& parsedInt : parsedInts) {
      // zero becomes `i64` when parsed, so truncate back down to minBitwidth
      parsedInt = parsedInt.isZero() ? parsedInt.trunc(minBitwidth) : parsedInt;
      maxWidth = std::max(maxWidth, parsedInt.getBitWidth());
    }
    for (auto& parsedInt : parsedInts) {
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
      return parser.emitError(parser.getNameLoc(),
                              "failed to parse scalar constant"),
             failure();
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

void ConstantOp::print(OpAsmPrinter& p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getOutput().getType());
}

// constant(c0) -> c0 mod q
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  Type type = getElementTypeOrSelf(getType());
  auto modType = dyn_cast<ModArithType>(type);
  if (!modType) return {};
  auto storageType = modType.getModulus().getType();

  // Retrieve the modulus value and its bit width
  APInt modulus = modType.getModulus().getValue();
  unsigned modBitWidth = modulus.getBitWidth();

  auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(adaptor.getValue());
  if (denseElementsAttr) {
    assert(isa<ShapedType>(getType()) &&
           "non-shaped type with shaped attribute");
    ShapedType shapedType = cast<ShapedType>(getType());
    SmallVector<APInt, 4> values;
    for (const auto value : denseElementsAttr.getValues<APInt>()) {
      values.push_back(value.zextOrTrunc(modBitWidth).urem(modulus));
    }
    // Have to use an integer type here because DenseIntElementsAttr
    // requires integer types.
    return DenseIntElementsAttr::get(
        RankedTensorType::get(shapedType.getShape(), storageType), values);
  }

  auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getValue());
  if (!intAttr) return {};

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
  return IntegerAttr::get(storageType, foldedVal);
}

namespace {
enum class ModOp { Add, Sub, Mul, Mac };

std::optional<APInt> getRawAPInt(Attribute attr) {
  if (!attr) return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getValue();
  }
  if (auto modAttr = dyn_cast<ModArithAttr>(attr)) {
    return modAttr.getValue().getValue();
  }
  return std::nullopt;
}

Attribute foldScalarModOp(ArrayRef<Attribute> operands,
                          ModArithType residueType, ModOp op,
                          StringRef opName) {
  auto lhs = getRawAPInt(operands[0]);
  auto rhs = getRawAPInt(operands[1]);
  if (!lhs || !rhs) return {};

  std::optional<APInt> acc = std::nullopt;
  if (op == ModOp::Mac) {
    if (operands.size() < 3) return {};
    acc = getRawAPInt(operands[2]);
    if (!acc) return {};
  }

  APInt modulus = residueType.getModulus().getValue();
  unsigned modBitWidth = modulus.getBitWidth();

  // Strict accumulation safety (1 extra bit or more)
  unsigned workWidth = 2 * modBitWidth + 1;

  APInt lw = lhs->zextOrTrunc(modBitWidth).zext(workWidth);
  APInt rw = rhs->zextOrTrunc(modBitWidth).zext(workWidth);
  APInt mw = modulus.zext(workWidth);

  APInt res;
  switch (op) {
    case ModOp::Add:
      res = lw + rw;
      break;
    case ModOp::Sub:
      res = lw + mw - rw.urem(mw);
      break;
    case ModOp::Mul:
      res = lw * rw;
      break;
    case ModOp::Mac: {
      APInt aw = acc->zextOrTrunc(modBitWidth).zext(workWidth);
      res = aw + (lw * rw);
      break;
    }
  }

  APInt foldedVal = res.urem(mw).trunc(modBitWidth);

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    llvm::dbgs() << "========================================\n";
    llvm::dbgs() << "  Folding Operation: " << opName << " (Limbwise)\n";
    llvm::dbgs() << "----------------------------------------\n";
    llvm::dbgs() << "  LHS     : " << *lhs << "\n";
    llvm::dbgs() << "  RHS     : " << *rhs << "\n";
    if (acc) {
      llvm::dbgs() << "  ACC     : " << *acc << "\n";
    }
    llvm::dbgs() << "  Modulus : " << modulus << "\n";
    llvm::dbgs() << "  Folded  : " << foldedVal << "\n";
    llvm::dbgs() << "========================================\n";
  });

  auto intAttr =
      IntegerAttr::get(residueType.getModulus().getType(), foldedVal);
  return ModArithAttr::get(residueType.getContext(), residueType, intAttr);
}

}  // namespace

Attribute AddOp::foldScalarResidue(ArrayRef<Attribute> operands,
                                   Type residueType) {
  auto modType = dyn_cast<ModArithType>(residueType);
  if (!modType) return {};
  return foldScalarModOp(operands, modType, ModOp::Add, "Add");
}
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return heir::foldLimbwise(getOperation(), adaptor.getOperands(),
                            getResult().getType());
}

Attribute SubOp::foldScalarResidue(ArrayRef<Attribute> operands,
                                   Type residueType) {
  auto modType = dyn_cast<ModArithType>(residueType);
  if (!modType) return {};
  return foldScalarModOp(operands, modType, ModOp::Sub, "Sub");
}
OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return heir::foldLimbwise(getOperation(), adaptor.getOperands(),
                            getResult().getType());
}

Attribute MulOp::foldScalarResidue(ArrayRef<Attribute> operands,
                                   Type residueType) {
  auto modType = dyn_cast<ModArithType>(residueType);
  if (!modType) return {};
  return foldScalarModOp(operands, modType, ModOp::Mul, "Mul");
}
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return heir::foldLimbwise(getOperation(), adaptor.getOperands(),
                            getResult().getType());
}

Attribute MacOp::foldScalarResidue(ArrayRef<Attribute> operands,
                                   Type residueType) {
  auto modType = dyn_cast<ModArithType>(residueType);
  if (!modType) return {};
  return foldScalarModOp(operands, modType, ModOp::Mac, "Mac");
}
OpFoldResult MacOp::fold(FoldAdaptor adaptor) {
  return heir::foldLimbwise(getOperation(), adaptor.getOperands(),
                            getResult().getType());
}

Operation* ModArithDialect::materializeConstant(OpBuilder& builder,
                                                Attribute value, Type type,
                                                Location loc) {
  if (auto limbwiseAttr = dyn_cast_if_present<LimbwiseAttrInterface>(value)) {
    return limbwiseAttr.materializeConstant(builder, loc, type);
  }
  if (auto modArithAttr = dyn_cast<ModArithAttr>(value)) {
    value = modArithAttr.getValue();
  }
  // TODO(#1759): support dense attributes
  auto intAttr = dyn_cast_if_present<IntegerAttr>(value);
  if (!intAttr) return nullptr;
  auto modType = dyn_cast_if_present<ModArithType>(type);
  if (!modType) return nullptr;
  auto op = mod_arith::ConstantOp::create(builder, loc, modType, intAttr);
  return op.getOperation();
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "lib/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}  // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<AddZero, AddAddConstant, AddSubConstantRHS, AddSubConstantLHS,
              AddMulNegativeOneRhs, AddMulNegativeOneLhs>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<SubZero, SubMulNegativeOneRhs, SubMulNegativeOneLhs,
              SubRHSAddConstant, SubLHSAddConstant, SubRHSSubConstantRHS,
              SubRHSSubConstantLHS, SubLHSSubConstantRHS, SubLHSSubConstantLHS,
              SubSubLHSRHSLHS>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MulZero, MulOne, MulMulConstant>(context);
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
