#include "lib/Transforms/LayoutPropagation/Utils.h"

#include <iostream>

#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

int64_t maxOfMaxes(ArrayRef<int64_t> d1, ArrayRef<int64_t> d2) {
  int64_t max = d1.front();
  for (int64_t i = 1; i < d1.size(); ++i) {
    max = std::max(max, d1[i]);
  }
  for (long i : d2) {
    max = std::max(max, i);
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

}  // namespace heir
}  // namespace mlir
