#ifndef LIB_DIALECT_ROTOM_UTILS_ROTOMTENSOREXTLAYOUTLOWERING_H_
#define LIB_DIALECT_ROTOM_UTILS_ROTOMTENSOREXTLAYOUTLOWERING_H_

#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

/// Lowers a Rotom `#rotom.layout` to the ISL map text used by
/// `tensor_ext.layout` (domain `i*`, range `ct`, `slot`).
struct RotomTensorExtLayoutLowering {
  static FailureOr<std::string> lowerToTensorExtIsl(LayoutAttr layout);
};

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_ROTOMTENSOREXTLAYOUTLOWERING_H_
