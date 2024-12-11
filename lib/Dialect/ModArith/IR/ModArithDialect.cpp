#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>
#include <optional>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define ModArithDialect,
// ModArithTypes, ModArithOps, ModArithAttributes
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

namespace mlir {
namespace heir {
namespace mod_arith {

class ModArithOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<ModArithType>([&](auto &modArithType) {
                     os << "Z";
                     os << modArithType.getModulus().getValue();
                     os << "_";
                     os << modArithType.getModulus().getType();
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void ModArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();

  addInterface<ModArithOpAsmDialectInterface>();
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

LogicalResult EncapsulateOp::verify() {
  auto modArithType = getResultModArithType(*this);
  auto integerType = getOperandIntegerType(*this);
  auto result = verifySameWidth(*this, modArithType, integerType);
  if (result.failed()) return result;
  return verifyModArithType(*this, getResultModArithType(*this));
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

LogicalResult ModSwitchOp::verify() {
  // auto srcModulus = getInput().getType().getModulus().getInt();
  // auto dstModulus = getOutput().getType().getModulus().getInt();

  auto srcModulus =
      cast<ModArithType>(getInput().getType()).getModulus().getInt();
  auto dstModulus =
      cast<ModArithType>(getOutput().getType()).getModulus().getInt();

  if (srcModulus >= dstModulus)
    return emitOpError() << "source modulus " << srcModulus
                         << " must be smaller than destination modulus "
                         << dstModulus << "\n"
                         << " This situations leads to information loss.";

  return success();
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
  APInt parsedValue(64, 0);
  Type parsedType;

  if (failed(parser.parseInteger(parsedValue))) {
    parser.emitError(parser.getCurrentLocation(),
                     "found invalid integer value");
    return failure();
  }

  if (parser.parseColon() || parser.parseType(parsedType)) return failure();

  auto modArithType = dyn_cast<ModArithType>(parsedType);
  if (!modArithType) return failure();

  auto outputBitWidth =
      modArithType.getModulus().getType().getIntOrFloatBitWidth();
  if (parsedValue.getActiveBits() > outputBitWidth)
    return parser.emitError(parser.getCurrentLocation(),
                            "constant value is too large for the modulus");

  auto intValue = IntegerAttr::get(modArithType.getModulus().getType(),
                                   parsedValue.trunc(outputBitWidth));
  result.addAttribute(
      "value", ModArithAttr::get(parser.getContext(), modArithType, intValue));
  result.addTypes(modArithType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  // getValue chain:
  // op's ModArithAttribute value
  //   -> ModArithAttribute's IntegerAttr value
  //   -> IntegerAttr's APInt value
  getValue().getValue().getValue().print(p.getStream(), true);
  p << " : ";
  p.printType(getOutput().getType());
}

LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> loc,
    ConstantOpAdaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(adaptor.getValue().getType());
  return success();
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
