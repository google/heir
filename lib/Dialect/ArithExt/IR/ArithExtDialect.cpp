#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"

#include <cassert>

#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define ArithExtDialect and
// ArithExtOps
#include "lib/Dialect/ArithExt/IR/ArithExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ArithExt/IR/ArithExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ArithExt/IR/ArithExtOps.cpp.inc"

namespace mlir {
namespace heir {
namespace arith_ext {

void ArithExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ArithExt/IR/ArithExtOps.cpp.inc"
      >();
}

/// Ensures that the underlying integer type is wide enough for the coefficient
template <typename OpType>
LogicalResult verifyArithExtOpMod(OpType op) {
  auto type =
      llvm::cast<IntegerType>(getElementTypeOrSelf(op.getResult().getType()));
  unsigned bitWidth = type.getWidth();
  unsigned modWidth = (op.getModulus() - 1).getActiveBits();
  if (modWidth > bitWidth)
    return op.emitOpError()
           << "underlying type's bitwidth must be at least as "
           << "large as the modulus bitwidth, but got " << bitWidth
           << " while modulus requires width " << modWidth << ".";
  if (!type.isUnsigned() && modWidth == bitWidth)
    emitWarning(op.getLoc())
        << "for signed (or signless) underlying types, the bitwidth of the "
           "underlying type must be at least as large as modulus bitwidth + "
           "1 (for the sign bit), but found "
        << bitWidth << " while modulus requires width " << modWidth << ".";
  return success();
}

LogicalResult AddOp::verify() { return verifyArithExtOpMod<AddOp>(*this); }

LogicalResult SubOp::verify() { return verifyArithExtOpMod<SubOp>(*this); }

LogicalResult MulOp::verify() { return verifyArithExtOpMod<MulOp>(*this); }

LogicalResult MacOp::verify() { return verifyArithExtOpMod<MacOp>(*this); }

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

  return success();
}

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir
