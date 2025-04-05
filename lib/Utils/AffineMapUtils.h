#ifndef LIB_UTILS_AFFINEMAPUTILS_H_
#define LIB_UTILS_AFFINEMAPUTILS_H_

#include <cstdint>
#include <numeric>

#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

namespace mlir {
namespace heir {

// Construct an identity permutation of size n
inline ::llvm::SmallVector<int64_t> identity(int64_t n) {
  ::llvm::SmallVector<int64_t> permutation;
  permutation.resize(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  return permutation;
}

// Populates `result` with a concrete permutation corresponding to the
// evaluation of `map` on the 1D input domain {1..rank}. Returns failure() if
// the map is not a permutation.
::llvm::LogicalResult makeExplicit1DMapping(
    ::mlir::AffineMap map, unsigned rank, ::llvm::SmallVector<int64_t> &result);

// Returns true if the materialized mapping is a permutation.
bool isPermutation(::llvm::ArrayRef<int64_t> materializedMapping);

template <typename T>
void printPermutation(::llvm::ArrayRef<int64_t> permutation, T &os) {
  for (int i = 0; i < permutation.size(); i++) {
    os << i << " -> " << permutation[i] << ", ";
    if (i % 10 == 9) {
      os << "\n";
    }
  }
  os << "\n";
}

bool isLayoutRowMajor(RankedTensorType inputType, RankedTensorType outputType,
                      const AffineMap &layout);

AffineMap getRowMajorLayoutMap(RankedTensorType inputType,
                               RankedTensorType outputType);

bool isLayoutSquatDiagonal(RankedTensorType inputType,
                           RankedTensorType outputType,
                           const AffineMap &layout);

AffineMap getDiagonalLayoutMap(RankedTensorType inputType,
                               RankedTensorType outputType);

template void printPermutation(::llvm::ArrayRef<int64_t>,
                               ::llvm::raw_ostream &);
template void printPermutation(::llvm::ArrayRef<int64_t>, ::mlir::Diagnostic &);

// Evaluate an affine map on statically known inputs and populate `results`.
void evaluateStatic(AffineMap map, ArrayRef<int64_t> values,
                    SmallVector<int64_t> &results);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_AFFINEMAPUTILS_H_
