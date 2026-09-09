#include "lib/Dialect/Rotom/IR/RotomOps.h"

#include <tuple>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Rotom/IR/RotomOps.cpp.inc"

namespace mlir {
namespace heir {
namespace rotom {

LogicalResult ConvertLayoutOp::verify() {
  if (getFrom().getN() != getTo().getN()) {
    return emitOpError() << "the from and to layouts must have the same "
                            "ciphertext size n; got "
                         << getFrom().getN() << " and " << getTo().getN();
  }
  return success();
}

LogicalResult ApplyRollOp::verify() {
  LayoutAttr from = getFrom();
  LayoutAttr to = getTo();
  if (from.getN() != to.getN()) {
    return emitOpError() << "the from and to layouts must have the same "
                            "ciphertext size n";
  }
  // A roll may swap the rolled piece with its partner, so the two piece lists
  // must be the same multiset.
  auto sortedDims = [](LayoutAttr layout) {
    SmallVector<DimAttr> dims;
    for (Attribute attr : layout.getDims()) dims.push_back(cast<DimAttr>(attr));
    llvm::sort(dims, [](DimAttr a, DimAttr b) {
      return std::tuple(a.getDim(), a.getSize(), a.getStride()) <
             std::tuple(b.getDim(), b.getSize(), b.getStride());
    });
    return dims;
  };
  if (sortedDims(from) != sortedDims(to)) {
    return emitOpError() << "the from and to layouts must have the same "
                            "pieces; an apply_roll adds one roll and may swap "
                            "the rolled piece with its partner";
  }
  ArrayRef<int64_t> fromRolls =
      from.getRolls() ? from.getRolls().asArrayRef() : ArrayRef<int64_t>();
  ArrayRef<int64_t> toRolls =
      to.getRolls() ? to.getRolls().asArrayRef() : ArrayRef<int64_t>();
  if (toRolls.size() != fromRolls.size() + 2 ||
      toRolls.take_front(fromRolls.size()) != fromRolls) {
    return emitOpError() << "the to layout must carry the from layout's rolls "
                            "plus exactly one more";
  }
  return success();
}

LogicalResult MatmulOp::verify() {
  const int64_t n = getCompute().getN();
  if (getLhsLayout().getN() != n || getRhsLayout().getN() != n ||
      getTo().getN() != n) {
    return emitOpError() << "all layouts must have the same ciphertext size n";
  }
  return success();
}

LogicalResult BsgsMatmulOp::verify() {
  if (getRollOperand() != 0 && getRollOperand() != 1) {
    return emitOpError() << "roll_operand must name lhs (0) or rhs (1)";
  }
  if (getBaby() <= 0 || getRollTargets() <= 0) {
    return emitOpError() << "baby and roll_targets must be positive";
  }
  if (getBaby() > getRollTargets()) {
    return emitOpError() << "baby extent exceeds the target count";
  }
  const int64_t n = getRolled().getN();
  if (getRollStride() <= 0 || getRollStride() >= n) {
    return emitOpError() << "roll_stride must be a rotation amount in [1, n)";
  }
  if (getLhsLayout().getN() != n || getRhsLayout().getN() != n ||
      getCompute().getN() != n || getTo().getN() != n) {
    return emitOpError() << "every layout must share the ciphertext size n";
  }
  return success();
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
