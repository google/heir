#ifndef THIRD_PARTY_HEIR_INCLUDE_UTILS_MEMREFUTILS_H_
#define THIRD_PARTY_HEIR_INCLUDE_UTILS_MEMREFUTILS_H_

#include "mlir/include/mlir/Dialect/Affine/Utils.h" // from @llvm-project

namespace mlir {

namespace heir {

// getFlattenedAccessIndex gets the flattened access index for MemRef access
// given the MemRef type's shape. Returns a std::nullopt if the indices are not
// constants (e.g. derived from inputs).
std::optional<uint64_t> getFlattenedAccessIndex(affine::MemRefAccess access,
                                                mlir::Type memRefType);

}  // namespace heir

}  // namespace mlir

#endif  // THIRD_PARTY_HEIR_INCLUDE_UTILS_MEMREFUTILS_H_
