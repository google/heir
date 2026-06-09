#ifndef LIB_ANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_H_
#define LIB_ANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_H_

#include <cstdint>

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"      // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

// For an individual encode op (a "site"), this struct records the computed
// offset of the (1 dimensional) memref that the preprocessed plaintexts should
// start at, and the number of plaintexts in that section of the memref.
struct SiteLayout {
  int64_t offset;
  int64_t size;
};

// An analysis that records all SiteLayouts by encode op, along with a total
// size of the lowered memref and a fast-fail flag in case invalid IR is
// detected. This analysis is used by preprocessing-to-memref to materialize the
// opaque preprocessing type to a memref.
class PreprocessingStorageLayoutAnalysis {
 public:
  explicit PreprocessingStorageLayoutAnalysis(Operation* op);

  bool isValid() const { return valid; }

  FailureOr<SiteLayout> getLayout(uint32_t siteId) const;

  FailureOr<int64_t> getTotalSize(Type type) const;
  const DenseMap<Type, int64_t>& getTotalSizes() const { return totalSizes; }

 private:
  DenseMap<uint32_t, SiteLayout> siteLayouts;
  DenseMap<Type, int64_t> totalSizes;
  bool valid = true;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_H_
