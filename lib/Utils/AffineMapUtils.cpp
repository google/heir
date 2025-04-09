#include "lib/Utils/AffineMapUtils.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"          // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult makeExplicit1DMapping(AffineMap map, unsigned rank,
                                    SmallVector<int64_t> &result) {
  if (map.getNumResults() != 1) return failure();
  if (map.getNumDims() != 1) return failure();
  if (map.getNumSymbols() != 0) return failure();

  OpBuilder b(map.getContext());
  result.resize(rank);

  SmallVector<Attribute> permInputs;
  permInputs.reserve(rank);
  for (size_t permInput = 0; permInput < rank; ++permInput) {
    permInputs.push_back(b.getIndexAttr(permInput));
  }

  llvm::copy(llvm::map_range(
                 permInputs,
                 [&](Attribute permInput) -> int64_t {
                   SmallVector<Attribute> results;
                   // AffineMap::constantFold is the mechanism to evaluate the
                   // affine map on statically known inputs.
                   if (failed(map.constantFold(permInput, results))) {
                     assert(false && "constant folding should never fail here");
                     return -1L;
                   }
                   return cast<IntegerAttr>(results[0]).getInt();
                 }),
             result.begin());

  return success();
}

bool isPermutation(ArrayRef<int64_t> materializedMapping) {
  DenseSet<int64_t> seen;
  for (int64_t i : materializedMapping) {
    if (i < 0 || i >= materializedMapping.size()) return false;
    if (!seen.insert(i).second) return false;
  }
  return true;
}

AffineMap getRowMajorLayoutMap(RankedTensorType inputType,
                               RankedTensorType outputType) {
  // Row-major forces (a, b, c) -> (a * dims[1] * dims[2] + b * dims[2] + c)
  // with an optional modulus of the output type size at the end.
  SmallVector<AffineExpr> dims;
  AffineExpr result;
  for (int dim = 0; dim < inputType.getRank(); ++dim) {
    dims.push_back(getAffineDimExpr(dim, inputType.getContext()));
  }

  for (int dim = dims.size() - 1; dim >= 0; --dim) {
    // iter 1: k
    // iter 2: k + size(k) * j
    // iter 3: k + size(k) * j + size(j) * size(k) * i
    result =
        result ? result + inputType.getDimSize(dim + 1) * dims[dim] : dims[dim];
  }

  AffineMap layout = AffineMap::get(dims.size(), 0, {result});
  return simplifyAffineMap(layout);
}

bool isLayoutRowMajor(RankedTensorType inputType, RankedTensorType outputType,
                      const AffineMap &layout) {
  // For now, only support 1D output; not sure what a "row major" multi-dim
  // output would mean (the trailing input dims collapsed on the trailing
  // output dim, with all other dims matching?).
  if (outputType.getRank() != 1) return false;

  AffineMap expected = getRowMajorLayoutMap(inputType, outputType);
  AffineMap expected2 =
      AffineMap::get(expected.getNumDims(), 0,
                     {expected.getResults()[0] % outputType.getNumElements()});
  auto simplified = simplifyAffineMap(layout);

  return (simplified == expected || simplified == expected2);
}

AffineMap getDiagonalLayoutMap(RankedTensorType inputType,
                               RankedTensorType outputType) {
  int64_t n = outputType.getDimSize(0);
  int64_t m = outputType.getDimSize(1);
  AffineExpr i, j;
  bindDims(inputType.getContext(), i, j);
  AffineMap layout =
      AffineMap::get(2, 0, {j % n, (i + j) % m}, inputType.getContext());
  return simplifyAffineMap(layout);
}

bool isLayoutSquatDiagonal(RankedTensorType inputType,
                           RankedTensorType outputType,
                           const AffineMap &layout) {
  // Squat diagonal forces (i, j) -> (j % n, (i+j) % m) where (n, m) are the
  // dimensions of the output matrix.
  if (outputType.getRank() != 2 || inputType.getRank() != 2) return false;
  auto simplified = simplifyAffineMap(layout);
  auto expected = getDiagonalLayoutMap(inputType, outputType);
  return simplified == expected;
}

inline Attribute getIndexAttr(MLIRContext *ctx, int64_t value) {
  return IntegerAttr::get(IndexType::get(ctx), value);
}

void evaluateStatic(AffineMap map, ArrayRef<int64_t> values,
                    SmallVector<int64_t> &results) {
  MLIRContext *context = map.getContext();
  SmallVector<Attribute> mapInputs = llvm::map_to_vector(
      values, [&](int64_t i) { return getIndexAttr(context, i); });

  // Evaluate the affine map on the inputs
  SmallVector<Attribute> foldResults;
  if (failed(map.constantFold(mapInputs, foldResults))) {
    assert(false && "constant folding should never fail here");
  }

  results.reserve(foldResults.size());
  for (Attribute attr : foldResults) {
    results.push_back(cast<IntegerAttr>(attr).getInt());
  }
}

}  // namespace heir
}  // namespace mlir
