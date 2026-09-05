#ifndef LIB_TRANSFORMS_PROPAGATEPADDING_PROPAGATEPADDING_H_
#define LIB_TRANSFORMS_PROPAGATEPADDING_PROPAGATEPADDING_H_

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"           // from @llvm-project

namespace mlir {
namespace heir {

// The name of the discardable op attribute the propagate-padding pass stamps
// on ops whose single result has known padding semantics.
constexpr const char kPaddingAttrName[] = "tensor_ext.padding";

#define GEN_PASS_DECL
#include "lib/Transforms/PropagatePadding/PropagatePadding.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/PropagatePadding/PropagatePadding.h.inc"

// Attaches PaddingSemanticsOpInterface external models to the upstream ops
// the propagate-padding pass understands (tensor.pad, linalg matmuls,
// elementwise arith, etc.).
void registerPaddingSemanticsInterfaces(DialectRegistry& registry);

// Returns the padding attribute stamped on `value`'s defining op by the
// propagate-padding pass, or null if absent. Consumers must treat null as
// "value is unpadded".
tensor_ext::PaddingAttr getPaddingInfo(Value value);

// Builds a 0/1 dense float constant over `type`: with `onesOutside` false,
// 1.0 exactly on the trailing-pad logical region (a mask); with it true,
// 1.0 exactly outside it (a pad-region pin).
DenseElementsAttr buildRegionIndicator(RankedTensorType type,
                                       ArrayRef<int64_t> logical,
                                       bool onesOutside, double oneValue = 1.0);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_PROPAGATEPADDING_PROPAGATEPADDING_H_
