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

// Returns an IntegerRelation that enforces a row-major layout for the given
// tensor type and number of slots. This is used for IntegerRelations that
// represent data layouts in ciphertexts. It expects that the number of domain
// variables match the rank of the tensor, and that there are two range
// variables representing the ciphertext index and slot index in that order.
presburger::IntegerRelation getRowMajorLayoutRelation(
    RankedTensorType tensorType, int64_t numSlots);

// Returns an IntegerRelation that represents a diagonalized layout for a matrix
// such that the ith diagonal of the matrix is in the ith row of the
// result. The number of rows of the input and output must match.
presburger::IntegerRelation getDiagonalLayoutRelation(
    RankedTensorType matrixType, int64_t ciphertextSize);

// Returns true if the given relation is a squat diagonal layout for the given
// matrix type and ciphertext semantic shape.
bool isRelationSquatDiagonal(RankedTensorType matrixType,
                             int64_t ciphertextSize,
                             presburger::IntegerRelation relation);

// Returns true if the given relation is a row-major layout for the given
// vector type and slot size.
bool isRelationRowMajor(RankedTensorType vectorType, int64_t numSlots,
                        presburger::IntegerRelation relation);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_UTILS_H_
