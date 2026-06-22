#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_

#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project

namespace mlir::heir::rotom {

// Relative HE op-cost weights, in "key-switch" units, that put rotations
// (conversions) and compute (adds/muls) on a common scale so they can be summed
// in a candidate's accumulated cost. See cost_model.json for the defaults and
// the rationale; these are temporary placeholders pending measured values.
struct RotomCostModel {
  int64_t rotation = 100;            // one ciphertext rotation (key-switch)
  int64_t ciphertextMultiply = 100;  // one ct x ct multiply (+ relinearization)
  int64_t add = 1;                   // one add / negligible, breaks ties
};

// The active cost model. Defaults to the struct's values (which mirror
// cost_model.json); on first use, if the ROTOM_COST_MODEL environment variable
// names a readable JSON file, that file overrides the defaults.
const RotomCostModel& getCostModel();
void setCostModel(const RotomCostModel& model);

// Parses a cost model from JSON (the schema in cost_model.json). Missing keys
// keep their default; returns nullopt only on malformed JSON.
std::optional<RotomCostModel> parseCostModel(llvm::StringRef json);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_COSTMODEL_H_
