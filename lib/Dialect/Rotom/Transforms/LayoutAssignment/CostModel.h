#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_

#include <cstdint>

namespace mlir::heir::rotom {

// TODO: Temporary placeholders pending measured values.
struct RotomCostModel {
  int64_t rotation = 100;            // one ciphertext rotation (key-switch)
  int64_t ciphertextMultiply = 100;  // one ct x ct multiply (+ relinearization)
  int64_t add = 1;                   // one add / negligible, breaks ties
  // Carrying cost per ciphertext of a value's chosen layout: a proxy for the
  // downstream price of a fat value (bootstrap placement wants compact
  // candidates -- every extra ciphertext is an extra future bootstrap).
  // Keeps the search from inflating ciphertext counts when an equally cheap
  // compact placement plus a rotation-only expansion exists.
  int64_t ciphertextCount = 100;
};

// The active cost model (currently always the defaults above).
const RotomCostModel& getCostModel();

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_
