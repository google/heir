#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"

namespace mlir::heir::rotom {

const RotomCostModel& getCostModel() {
  static const RotomCostModel model;
  return model;
}

}  // namespace mlir::heir::rotom
