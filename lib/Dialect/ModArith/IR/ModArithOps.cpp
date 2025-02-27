#include "lib/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir {
namespace heir {
namespace mod_arith {

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

// copied from ArithOps.cpp
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // addi(x, 0) -> x
  if (auto modArithAttr =
          mlir::dyn_cast_or_null<ModArithAttr>(adaptor.getRhs())) {
    if (modArithAttr.getValue().getValue() == 0) return getLhs();
  }

  // addi(subi(a, b), b) -> a
  if (auto sub = getLhs().getDefiningOp<SubOp>())
    if (getRhs() == sub.getRhs()) return sub.getLhs();

  // addi(b, subi(a, b)) -> a
  if (auto sub = getRhs().getDefiningOp<SubOp>())
    if (getLhs() == sub.getRhs()) return sub.getLhs();

  // TODO(#1216): fold constant a + b
  // See ArithOps.td for detail
  return {};
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
