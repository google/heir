#include "lib/Utils/Layout/ConvolutionTestUtil.h"

#include <cstdint>

#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

FailureOr<presburger::IntegerRelation>
get2dConvChwFchwFilterDiagonalizedRelation(RankedTensorType filterType,
                                           RankedTensorType dataType,
                                           ArrayRef<int64_t> strides,
                                           int64_t padding,
                                           int64_t ciphertextSize,
                                           bool interchangeRows) {
  auto expandedFilterRelation =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);
  // Permutate the rows of the matrix to minimize the number of non-zero
  // diagonals.
  if (interchangeRows) {
    int64_t dataRowSize = dataType.getDimSize(2);
    int64_t dataColSize = dataType.getDimSize(3);
    int64_t filterRowSize = filterType.getDimSize(2);
    int64_t filterColSize = filterType.getDimSize(3);
    int64_t strideRow = strides[0];
    int64_t strideCol = strides[1];
    int64_t outputH =
        (dataRowSize + 2 * padding - filterRowSize) / strideRow + 1;
    int64_t outputW =
        (dataColSize + 2 * padding - filterColSize) / strideCol + 1;

    int64_t inputChannels = dataType.getDimSize(1);
    RankedTensorType singleFilterType = RankedTensorType::get(
        {filterRowSize, filterColSize}, filterType.getElementType());
    RankedTensorType singleDataType = RankedTensorType::get(
        {dataRowSize, dataColSize}, dataType.getElementType());
    auto singleResultType = get2dConvFilterExpandedType(
        singleFilterType, singleDataType, padding, strides);
    int64_t totalColSize = singleResultType.getDimSize(1);
    int64_t maxCol = inputChannels * totalColSize;

    auto rowInterchangeRelation = get2dConvRowInterchangeRelation(
        filterType.getDimSize(0), outputH, outputW, strides[0]);
    rowInterchangeRelation.appendVar(presburger::VarKind::Domain);
    rowInterchangeRelation.appendVar(presburger::VarKind::Range);
    addBounds(
        rowInterchangeRelation,
        rowInterchangeRelation.getVarKindOffset(presburger::VarKind::Domain) +
            1,
        0, maxCol - 1);
    addBounds(
        rowInterchangeRelation,
        rowInterchangeRelation.getVarKindOffset(presburger::VarKind::Range) + 1,
        0, maxCol - 1);
    addConstraint(
        rowInterchangeRelation,
        {{rowInterchangeRelation.getVarKindOffset(presburger::VarKind::Domain) +
              1,
          -1},
         {rowInterchangeRelation.getVarKindOffset(presburger::VarKind::Range) +
              1,
          1}},
        /*equality=*/true);

    auto diagonalizedInterchange =
        diagonalize2dMatrix(rowInterchangeRelation, filterType, ciphertextSize);
    if (failed(diagonalizedInterchange)) return failure();

    expandedFilterRelation.compose(diagonalizedInterchange.value());
    return expandedFilterRelation;
  }
  auto res =
      diagonalize2dMatrix(expandedFilterRelation, filterType, ciphertextSize);
  return res;
}

}  // namespace heir
}  // namespace mlir
