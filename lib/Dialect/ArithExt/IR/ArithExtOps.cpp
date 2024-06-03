#include "lib/Dialect/ArithExt/IR/ArithExtOps.h"

#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith_ext {

LogicalResult BarrettReduceOp::verify() {
  auto inputType = getInput().getType();
  unsigned bitWidth;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    bitWidth = tensorType.getElementTypeBitWidth();
  } else if (auto integerType = dyn_cast<IntegerType>(inputType)) {
    bitWidth = integerType.getWidth();
  }
  auto cmod = APInt(64, getModulus());
  auto expectedBitWidth = (cmod - 1).getActiveBits();
  if (bitWidth < expectedBitWidth || 2 * expectedBitWidth < bitWidth) {
    return emitOpError()
           << "input bitwidth is required to be in the range [w, 2w], where w "
              "is the smallest bit-width that contains the range [0, modulus). "
              "Got "
           << bitWidth << " but w is " << expectedBitWidth << ".";
  }

  return success();
}

ConstantIntRanges initialNormalisedRange(uint64_t q) {
  return ConstantIntRanges::fromUnsigned(APInt(64, 0), APInt(64, q));
}

void BarrettReduceOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                        SetIntRangeFn setResultRange) {
  setResultRange(getResult(), initialNormalisedRange(2 * getModulus()));
}

void NormalisedOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), initialNormalisedRange(getQ()));
}

void SubIfGEOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                  SetIntRangeFn setResultRange) {
  auto lhsRange = inputRanges[0];
  auto rhsRange = inputRanges[1];
  setResultRange(getResult(),
                 ConstantIntRanges::fromUnsigned(
                     lhsRange.umin() - rhsRange.umax(), lhsRange.umax()));
}

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir
