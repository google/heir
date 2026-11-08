#ifndef LIB_DIALECT_JAXITEWORD_IR_LEVELUTILS_H_
#define LIB_DIALECT_JAXITEWORD_IR_LEVELUTILS_H_

#include <cstdint>

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

/// Returns the largest LWE modulus-chain current value in the enclosing
/// module.
int64_t getMaxCiphertextCurrent(Operation* op);

/// Converts an LWE modulus-chain current value to an offset from CROSS's
/// ctx.max_level. extraOffset supports operations whose accessor level differs
/// from the level carried by their ciphertext operand.
int64_t getCrossLevelOffset(int64_t current, Operation* op,
                            int64_t extraOffset = 0);
int64_t getCrossLevelOffset(Value ciphertext, Operation* op,
                            int64_t extraOffset = 0);

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_JAXITEWORD_IR_LEVELUTILS_H_
