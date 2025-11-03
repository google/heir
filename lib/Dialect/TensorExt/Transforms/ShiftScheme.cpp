#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"

#include <cassert>
#include <cstdint>

#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

#define DEBUG_TYPE "shift-scheme"

namespace mlir {
namespace heir {
namespace tensor_ext {

SmallVector<int64_t> defaultShiftOrder(int64_t n) {
  SmallVector<int64_t> result;
  int64_t maxLog2 = APInt(64, n).getActiveBits();
  if (isPowerOfTwo(n)) maxLog2 -= 1;
  for (int64_t i = 0; i < maxLog2; i++) result.push_back(1 << i);
  return result;
}

// Convert an input->output index mapping to a canonical left-shift amount for
// a given tensor size.
// Example: 1 -> 13 with a 64-size tensor should produce a rotation of 52
// Example: 13 -> 1 with a 64-size tensor should produce a rotation of 12
inline int64_t normalizeShift(int64_t input, int64_t output,
                              int64_t tensorSize) {
  int64_t shift = (output - input) % tensorSize;
  shift = -shift;  // Account for leftward rotations
  if (shift < 0) {
    shift += tensorSize;
  }
  return shift;
}

int64_t ShiftStrategy::getVirtualShift(const CtSlot& source,
                                       const CtSlot& target) const {
  int64_t sourceIndex = source.ct * ciphertextSize + source.slot;
  int64_t targetIndex = target.ct * ciphertextSize + target.slot;
  return normalizeShift(sourceIndex, targetIndex, virtualCiphertextSize);
}

void ShiftStrategy::evaluate(const Mapping& mapping) {
  // First compute the virtual shifts needed for each source slot
  SmallVector<SourceShift> sourceShifts;
  sourceShifts.reserve(mapping.size());
  for (const auto& [target, source] : mapping.getTargetToSource()) {
    int64_t shift = getVirtualShift(source, target);
    sourceShifts.push_back({source, shift});
  }

  // Compute the corresponding table of positions after each rotation,
  // akin to the table in Figure 3 of the Vos-Vos-Erkin paper, including the
  // first column of values that have not yet been rotated.
  rounds.reserve(shiftOrder.size() + 1);
  ShiftRound initialRound;
  for (const SourceShift& ss : sourceShifts) {
    initialRound.positions[ss] = ss.source;
    initialRound.rotationAmount = 0;
  }
  rounds.push_back(initialRound);

  for (auto rotationAmount : shiftOrder) {
    auto lastRoundPositions = rounds.back().positions;
    DenseMap<SourceShift, CtSlot> currentRoundPosns;

    for (const SourceShift& key : sourceShifts) {
      assert(lastRoundPositions.contains(key) &&
             "Expected to find source in last round positions");
      CtSlot currentPos = lastRoundPositions[key];
      int64_t currentVirtualSlot =
          currentPos.ct * ciphertextSize + currentPos.slot;

      CtSlot nextPosition = currentPos;
      if (rotationAmount & key.shift) {
        currentVirtualSlot =
            (currentVirtualSlot - rotationAmount + virtualCiphertextSize) %
            virtualCiphertextSize;
        nextPosition = CtSlot{currentVirtualSlot / ciphertextSize,
                              currentVirtualSlot % ciphertextSize};
      }
      currentRoundPosns[key] = nextPosition;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After rotation " << rotationAmount << ":\n";
      for (const auto& [ss, pos] : currentRoundPosns) {
        llvm::dbgs() << "  (" << ss.source.ct << "," << ss.source.slot << ")["
                     << ss.shift << "] -> (" << pos.ct << "," << pos.slot << ")"
                     << "\n";
      }
    });

    rounds.push_back({currentRoundPosns, rotationAmount});
  }
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
