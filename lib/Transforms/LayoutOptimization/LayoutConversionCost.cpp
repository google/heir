#include "LayoutConversionCost.h"

#include <cstdint>

#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/TransformUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

#define DEBUG_TYPE "layout-conversion-cost"

namespace mlir {
namespace heir {

using Cost = int64_t;

// An unset value of a permutation as it's being built up.
static constexpr int kUnset = -1;

SmallVector<int64_t> createPermutation(Value value, int64_t slots,
                                       tensor_ext::LayoutAttr fromLayout,
                                       tensor_ext::LayoutAttr toLayout) {
  Type dataSemanticType = value.getType();
  SmallVector<int64_t> permutation(slots, kUnset);

  std::set<int64_t> uniqueRotations = {};

  // see
  // lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.cpp::ConvertConvertLayout
  // for a full explanation of the algorithm
  int64_t minUnusedTarget = 0;
  int64_t minUnusedInput = 0;
  while (minUnusedInput != -1) {
    IndexTupleConsumer evaluateNextIndex =
        [&](const std::vector<int64_t> &indices) {
          SmallVector<int64_t> fromResults;
          SmallVector<int64_t> toResults;
          evaluateStatic(fromLayout.getMap(), indices, fromResults);
          evaluateStatic(toLayout.getMap(), indices, toResults);
          int64_t input =
              (minUnusedInput + fromResults[fromResults.size() - 1]) % slots;
          int64_t output =
              (minUnusedTarget + toResults[toResults.size() - 1]) % slots;
          permutation[input] = output;
        };

    SmallVector<int64_t> dataSemanticShape;
    if (auto tensorTy = dyn_cast<RankedTensorType>(dataSemanticType)) {
      dataSemanticShape = SmallVector<int64_t>(tensorTy.getShape());
    } else {
      // assumed to be a scalar
      dataSemanticShape = {1};
    }

    LLVM_DEBUG(llvm::dbgs() << "dataSemanticShape: \n"; {
      for (int64_t val : dataSemanticShape) {
        llvm::dbgs() << val << ' ';
      };
    } llvm::dbgs() << "\n";);

    iterateIndices(dataSemanticShape, evaluateNextIndex);
    minUnusedTarget = getMinUnusedTarget(permutation);
    minUnusedInput = getMinUnusedInput(permutation);
  }

  LLVM_DEBUG(llvm::dbgs() << "my permutation: \n"; {
    for (int64_t val : permutation) {
      llvm::dbgs() << val << ' ';
    };
  } llvm::dbgs() << "\n";);
  return permutation;
}

Cost computeCostOfLayoutConversion(Value value, int64_t slots,
                                   tensor_ext::LayoutAttr fromLayout,
                                   tensor_ext::LayoutAttr toLayout) {
  if (fromLayout == toLayout) {
    LLVM_DEBUG(llvm::dbgs() << "Layouts are the same, conversion cost is 0\n";);
    return 0;
  }
  SmallVector<int64_t> permutation =
      createPermutation(value, slots, fromLayout, toLayout);
  FrozenVector<int64_t> permKey = FrozenVector<int64_t>(std::move(permutation));

  tensor_ext::VosVosErkinShiftNetworks shiftNetwork{slots};
  ArrayRef<tensor_ext::RotationGroup> vveRotationGroups =
      shiftNetwork.computeShiftNetwork(permKey);

  int64_t maxRotations = 0;
  for (const tensor_ext::RotationGroup &group : vveRotationGroups) {
    tensor_ext::ShiftStrategy rotationStrategy;
    rotationStrategy.evaluate(permKey, group);
    int64_t numRounds = rotationStrategy.getRounds().size();
    LLVM_DEBUG(llvm::dbgs() << "Number of rounds in this group is " << numRounds
                            << "\n";);
    if (numRounds > maxRotations) {
      maxRotations = numRounds;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Estimated cost is " << maxRotations << "\n";);
  return maxRotations;
}

}  // namespace heir
}  // namespace mlir
