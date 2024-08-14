#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>

#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define ModArithDialect and
// ModArithOps
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

namespace mlir {
namespace heir {
namespace mod_arith {

void ModArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();
}

/// Ensures that the underlying integer type is wide enough for the coefficient
template <typename OpType>
LogicalResult verifyModArithOpMod(OpType op) {
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

LogicalResult AddOp::verify() { return verifyModArithOpMod<AddOp>(*this); }

LogicalResult SubOp::verify() { return verifyModArithOpMod<SubOp>(*this); }

LogicalResult MulOp::verify() { return verifyModArithOpMod<MulOp>(*this); }

LogicalResult MacOp::verify() { return verifyModArithOpMod<MacOp>(*this); }

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

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
