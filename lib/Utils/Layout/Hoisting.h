#ifndef LIB_UTILS_LAYOUT_HOISTING_H_
#define LIB_UTILS_LAYOUT_HOISTING_H_

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Hoist the conversion of a vector layout through a matrix-vector multiply
// operation.
//
// Returns a new layout for the matrix argument of the matvec op.
//
// Note: this function requires the assumption that the chosen packing for the
// vector (and the corresponding matvec kernel) packs the vector into a single
// ciphertext.
presburger::IntegerRelation hoistConversionThroughMatvec(
    const presburger::IntegerRelation& matrixLayout,
    const presburger::IntegerRelation& fromVecLayout,
    const presburger::IntegerRelation& toVecLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_HOISTING_H_
