#ifndef THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_UTILS_H_
#define THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_UTILS_H_

#include "mlir/include/mlir/Dialect/Affine/Utils.h" // from @llvm-project

namespace mlir {
namespace heir {

// Extract statically constant input indices from a MemRefAccess to a vector of
// indexes per dimension. Returns a std::nullopt if the indices are not
// constants (e.g. derived from inputs).
std::optional<std::vector<uint64_t>> materialize(affine::MemRefAccess access);

// Extract and flatten the index of a MemRefAccess to the corresponding index in
// a 1-dimensional flattened array. Returns a std::nullopt if the indices are
// not constants (e.g. derived from inputs).
std::optional<uint64_t> getFlattenedAccessIndex(affine::MemRefAccess access,
                                                mlir::Type memRefType);

// Unflatten a flattened index for a memref with strided and offset metadata.
llvm::SmallVector<int64_t> unflattenIndex(int64_t index,
                                          const llvm::ArrayRef<int64_t> strides,
                                          int64_t offset);

// Flatten an index multiset for a memref with strided and offset metadata.
int64_t flattenIndex(const llvm::ArrayRef<int64_t> indices,
                     const llvm::ArrayRef<int64_t> strides, int64_t offset);

}  // namespace heir
}  // namespace mlir

#endif  // THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_UTILS_H_
