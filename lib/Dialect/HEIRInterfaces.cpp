// Block clang-format from thinking the header is unused
// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
#include "lib/Dialect/HEIRInterfaces.cpp.inc"

// Helper function to find all the relin basis in the function
SmallVector<int32_t> findAllRelinBasis(func::FuncOp op) {
  std::set<int32_t> distinctRelinBasis;
  op.walk([&](RelinearizeOpInterface relinOp) {
    for (auto basis : relinOp.getFromBasis()) {
      distinctRelinBasis.insert(basis);
    }
    return WalkResult::advance();
  });
  SmallVector<int32_t> relinBasis(distinctRelinBasis.begin(),
                                  distinctRelinBasis.end());
  return relinBasis;
}

// Helper function to find all the rotation indices in the function
SmallVector<int64_t> findAllRotIndices(func::FuncOp op) {
  std::set<int64_t> distinctRotIndices;
  op.walk([&](RotateOpInterface rotOp) {
    distinctRotIndices.insert(rotOp.getRotationOffset());
    return WalkResult::advance();
  });
  SmallVector<int64_t> rotIndicesResult(distinctRotIndices.begin(),
                                        distinctRotIndices.end());
  return rotIndicesResult;
}

}  // namespace heir
}  // namespace mlir
