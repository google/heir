#include "lib/Dialect/JaxiteWord/IR/LevelUtils.h"

#include <algorithm>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

int64_t getMaxCiphertextCurrent(Operation* op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return 0;
  int64_t maxCurrent = 0;
  module.walk([&](Operation* inner) {
    for (Value result : inner->getResults()) {
      if (auto type = dyn_cast<lwe::LWECiphertextType>(result.getType())) {
        maxCurrent =
            std::max(maxCurrent,
                     static_cast<int64_t>(type.getModulusChain().getCurrent()));
      }
    }
    for (Value operand : inner->getOperands()) {
      if (auto type = dyn_cast<lwe::LWECiphertextType>(operand.getType())) {
        maxCurrent =
            std::max(maxCurrent,
                     static_cast<int64_t>(type.getModulusChain().getCurrent()));
      }
    }
  });
  return maxCurrent;
}

int64_t getCrossLevelOffset(int64_t current, Operation* op,
                            int64_t extraOffset) {
  return getMaxCiphertextCurrent(op) - current + extraOffset;
}

int64_t getCrossLevelOffset(Value ciphertext, Operation* op,
                            int64_t extraOffset) {
  auto type = cast<lwe::LWECiphertextType>(ciphertext.getType());
  return getCrossLevelOffset(type.getModulusChain().getCurrent(), op,
                             extraOffset);
}

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
