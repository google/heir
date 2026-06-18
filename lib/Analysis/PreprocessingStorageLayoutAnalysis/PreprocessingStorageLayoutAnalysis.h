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

  // Returns the layout for a given site ID and element type.
  // The layout is relative to the element type's flat memref.
  FailureOr<SiteLayout> getLayout(Type type, uint32_t siteId) const;

  FailureOr<int64_t> getTotalSize(Type type) const;
  const DenseMap<Type, int64_t>& getTotalSizes() const { return totalSizes; }

 private:
  // Layouts are grouped hierarchically: Type -> site_id -> SiteLayout.
  // This allows computing relative offsets per Type, which is useful for
  // backends that split the storage object into separate memrefs by type.
  // Note that in some backends (like OpenFHE), these separate types may be
  // combined back into a single type (e.g., if they are lowered to a single
  // ciphertext type).
  //
  // Example:
  // If we have two types (i32, f32) and six globally unique site_ids:
  //   i32: site 0 (size 1), site 2 (size 3), site 4 (size 4)
  //   f32: site 1 (size 2), site 3 (size 4), site 5 (size 9)
  //
  // The resulting siteLayouts map will be:
  //   {
  //     i32: {
  //       0: {offset: 0, size: 1},
  //       2: {offset: 1, size: 3},
  //       4: {offset: 4, size: 4}
  //     },
  //     f32: {
  //       1: {offset: 0, size: 2},
  //       3: {offset: 2, size: 4},
  //       5: {offset: 6, size: 9}
  //     }
  //   }
  //
  // And totalSizes will be:
  //   { i32: 8, f32: 15 }
  DenseMap<Type, DenseMap<uint32_t, SiteLayout>> siteLayouts;
  DenseMap<Type, int64_t> totalSizes;
  bool valid = true;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_PREPROCESSINGSTORAGELAYOUTANALYSIS_H_
