#ifndef LIB_DIALECT_ROTOM_UTILS_ROTOMTENSOREXTLAYOUTLOWERING_H_
#define LIB_DIALECT_ROTOM_UTILS_ROTOMTENSOREXTLAYOUTLOWERING_H_

#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

/// A kernel-plan encoding composed into the materialized relation AFTER the
/// layout's own rolls: `fromPiece`'s axis index is rewritten to
/// (idx + multiplier * digit(byPiece)) mod extent. This is the
/// baby-step/giant-step giant pre-rotation -- a kernel-schedule shift by a
/// MULTIPLE of a digit, deliberately not layout vocabulary (layout rolls
/// shift by exactly the partner index). The plan that wants the packing
/// carries it; layouts stay unit-step and alignable endpoint-for-endpoint.
struct PreRotation {
  int64_t fromPiece;   // dims position whose axis is pre-rotated
  int64_t byPiece;     // dims position whose digit scales the shift
  int64_t multiplier;  // signed; positive shifts WITH the digit
};

/// Lowers a Rotom `#rotom.layout` to the ISL map text used by
/// `tensor_ext.layout` (domain `i*`, range `ct`, `slot`), optionally
/// composing a plan-level pre-rotation into the relation.
struct RotomTensorExtLayoutLowering {
  static FailureOr<std::string> lowerToTensorExtIsl(LayoutAttr layout);
  static FailureOr<std::string> lowerToTensorExtIsl(
      LayoutAttr layout, const PreRotation& preRotation);
};

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_ROTOMTENSOREXTLAYOUTLOWERING_H_
