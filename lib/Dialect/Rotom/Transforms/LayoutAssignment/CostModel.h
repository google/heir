#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_

#include <cstdint>

namespace mlir::heir::rotom {

// Relative HE op-cost weights, in "key-switch" units, that put rotations
// (conversions) and compute (adds/muls) on a common scale so they can be summed
// in a candidate's accumulated cost. These mirror cost_model.json and are
// temporary placeholders pending values measured for the target scheme.
struct RotomCostModel {
  int64_t rotation = 100;            // one ciphertext rotation (key-switch)
  int64_t ciphertextMultiply = 100;  // one ct x ct multiply (+ relinearization)
  int64_t add = 1;                   // one add / negligible, breaks ties
};

// The active cost model (currently always the defaults above).
const RotomCostModel& getCostModel();

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_
