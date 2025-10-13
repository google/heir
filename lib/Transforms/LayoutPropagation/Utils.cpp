#include "lib/Transforms/LayoutPropagation/Utils.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

int64_t maxOfMaxes(ArrayRef<int64_t> d1, ArrayRef<int64_t> d2) {
  int64_t max = d1.front();
  for (int64_t di : d1) {
    max = std::max(max, di);
  }
  for (int64_t di : d2) {
    max = std::max(max, di);
  }
  return max;
}

SmallVector<int64_t> shiftByInserted(ArrayRef<int64_t> dims,
                                     ArrayRef<int64_t> inserts,
                                     bool increment) {
  SmallVector<int64_t> result;
  SmallVector<int64_t> sortedDims(dims);
  SmallVector<int64_t> sortedInserts(inserts);
  llvm::sort(sortedDims);
  llvm::sort(sortedInserts);

  int64_t shift = 0;
  auto dimIt = sortedDims.begin(), insertIt = sortedInserts.begin();
  while (dimIt != sortedDims.end()) {
    auto materializedDim = *dimIt + (increment ? shift : -shift);
    if (insertIt < sortedInserts.end() && *insertIt <= materializedDim) {
      ++insertIt;
      ++shift;
    } else {
      result.push_back(materializedDim);
      ++dimIt;
    }
  }

  return result;
}

SmallVector<int64_t> shiftByRemoved(ArrayRef<int64_t> dims,
                                    ArrayRef<int64_t> removed) {
  return shiftByInserted(dims, removed, false);
}

LayoutAttr convertLayoutForReduce(LayoutAttr inputLayout,
                                  ArrayRef<int64_t> dimsToReduce) {
  std::unique_ptr<presburger::IntegerRelation> clonedRelation =
      inputLayout.getIntegerRelation().clone();

  auto offset = clonedRelation->getVarKindOffset(presburger::VarKind::Domain);
  for (int dim : llvm::reverse(dimsToReduce)) {
    // Set the dim to reduce equal to 0.
    auto dimIndex = offset + dim;
    assert(clonedRelation->getVarKindAt(dimIndex) ==
           presburger::VarKind::Domain);
    clonedRelation->setAndEliminate(dimIndex, 0);
  }

  MLIRContext* context = inputLayout.getContext();
  return LayoutAttr::getFromIntegerRelation(context,
                                            std::move(*clonedRelation));
}

}  // namespace heir
}  // namespace mlir
