#ifndef LIB_DIALECT_ROTOM_UTILS_LAYOUTCONVERSION_H_
#define LIB_DIALECT_ROTOM_UTILS_LAYOUTCONVERSION_H_

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// Estimates the cost of converting `from` to `to`, for the layout search.
// Computed from the piece and roll structure in O(#pieces).
struct ConversionEstimate {
  int64_t rotations = 0;
  int64_t masks = 0;
  int64_t accumulates = 0;
  // False when planLayoutConversion cannot express the conversion. The search
  // must not choose such a step: the lowering leaves the op in place and the
  // circuit it belongs to is silently lost.
  bool lowerable = true;
};
ConversionEstimate estimateConversionCost(LayoutAttr from, LayoutAttr to);

// One step of an explicit layout expansion: take ciphertext `sourceCt` of the
// source, rotate its slots left by `shift`, keep only `targetSlots`, and
// accumulate into ciphertext `targetCt` of the target. A step whose
// `targetSlots` cover all n slots needs no mask (a plain copy when the shift
// is also zero).
struct LayoutConversionStep {
  int64_t targetCt;
  int64_t sourceCt;
  int64_t shift;
  llvm::SmallVector<int64_t> targetSlots;
};

// Computes the rotate/mask/accumulate steps that convert `from` to `to`.
struct ReplicationFill {
  int64_t stride;
  int64_t extent;
};

// The steps that place one copy of the data, then the replications that fill
// the rest. Both the emission and the price come from this one plan, so they
// cannot disagree.
struct ConversionPlan {
  llvm::SmallVector<LayoutConversionStep> steps;
  llvm::SmallVector<ReplicationFill> fills;
};

FailureOr<ConversionPlan> planLayoutConversion(LayoutAttr from, LayoutAttr to);

// A plan that is a roll by the ciphertext piece.
struct BsgsSchedule {
  int64_t stride;
  int64_t targets;
  bool negative;
};
// The roll read off the layouts, with no plan to materialize.
std::optional<BsgsSchedule> bsgsScheduleOpt(LayoutAttr from, LayoutAttr to);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_LAYOUTCONVERSION_H_
