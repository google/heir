#ifndef LIB_UTILS_LAYOUT_UTILS_H_
#define LIB_UTILS_LAYOUT_UTILS_H_

#include <cstdint>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// Adds a new local variable q to the relation that represents expr % modulus.
// Returns the index of the new local variable in the relation.
unsigned int addModConstraint(presburger::IntegerRelation& result,
                              ArrayRef<int64_t> exprs, int64_t modulus);

// Adds a constraint to the given result relation that enforces a row-major
// layout for the given tensor type and number of slots. This is used for
// IntegerRelations that represent data layouts in ciphertexts. It expects that
// the number of domain variables match the rank of the tensor, and that there
// are two range variables representing the ciphertext index and slot index in
// that order.
void addRowMajorConstraint(presburger::IntegerRelation& result,
                           RankedTensorType tensorType, int64_t numSlots);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_UTILS_H_
