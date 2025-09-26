#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_

/// An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
/// http://dx.doi.org/10.1007/978-3-031-17140-6_20
///
/// This implements a version of the algorithm that supports arbitrary mappings
/// across multi-ciphertexts, including replication.

#include <cstdint>
#include <utility>

#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DECL_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

// Cf. https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20
// for an explanation of the algorithm.
class VosVosErkinShiftNetworks {
  using CacheKey = std::pair<Mapping, FrozenVector<int64_t>>;

 public:
  VosVosErkinShiftNetworks() = default;

  // Computes a partition of the slot indices of a ciphertext into
  // RotationGroups that are compatible with respect to the target permutation.
  // Each RotationGroup corresponds to a set of indices that should be rotated
  // together via power-of-two rotations.
  //
  // The returned ArrayRef is owned by this VosVosErkinShiftNetworks instance.
  // The resulting set of rotation groups are is cached, and the cache is used
  // on further calls to avoid recomputing the shift network.
  //
  // The default shiftOrder is LSB to MSB, i.e. 1, 2, 4, 8, ...
  ShiftScheme findShiftScheme(const Mapping& mapping,
                              ArrayRef<int64_t> shiftOrder = {});

 private:
  DenseMap<CacheKey, ShiftScheme> schemeCache;
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_
