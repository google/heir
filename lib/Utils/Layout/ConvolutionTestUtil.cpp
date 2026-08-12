#include "lib/Utils/Layout/ConvolutionTestUtil.h"

#include <cstdint>
#include <vector>

#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

int64_t convOutputExtent(int64_t dataExtent, int64_t filterExtent,
                         int64_t stride, int64_t padding) {
  return (dataExtent + 2 * padding - filterExtent) / stride + 1;
}

ConvTensor4D deterministicConvFilter(int64_t outputChannels,
                                     int64_t inputChannels, int64_t filterH,
                                     int64_t filterW) {
  ConvTensor4D filter(
      outputChannels,
      std::vector<std::vector<std::vector<int>>>(
          inputChannels, std::vector<std::vector<int>>(
                             filterH, std::vector<int>(filterW, 0))));
  for (int64_t f = 0; f < outputChannels; ++f) {
    for (int64_t c = 0; c < inputChannels; ++c) {
      for (int64_t fh = 0; fh < filterH; ++fh) {
        for (int64_t fw = 0; fw < filterW; ++fw) {
          filter[f][c][fh][fw] =
              (int)((f * 37 + c * 11 + fh * 5 + fw * 3) % 17) + 1;
        }
      }
    }
  }
  return filter;
}

ConvTensor4D reference2dConv(const ConvTensor4D& data,
                             const ConvTensor4D& filter, int64_t stride,
                             int64_t padding) {
  int64_t inputChannels = data[0].size();
  int64_t dataH = data[0][0].size();
  int64_t dataW = data[0][0][0].size();
  int64_t outputChannels = filter.size();
  int64_t filterH = filter[0][0].size();
  int64_t filterW = filter[0][0][0].size();
  int64_t outputH = convOutputExtent(dataH, filterH, stride, padding);
  int64_t outputW = convOutputExtent(dataW, filterW, stride, padding);

  ConvTensor4D result(
      1, std::vector<std::vector<std::vector<int>>>(
             outputChannels, std::vector<std::vector<int>>(
                                 outputH, std::vector<int>(outputW, 0))));
  for (int64_t f = 0; f < outputChannels; ++f) {
    for (int64_t oh = 0; oh < outputH; ++oh) {
      for (int64_t ow = 0; ow < outputW; ++ow) {
        int sum = 0;
        for (int64_t c = 0; c < inputChannels; ++c) {
          for (int64_t fh = 0; fh < filterH; ++fh) {
            for (int64_t fw = 0; fw < filterW; ++fw) {
              int64_t h = oh * stride - padding + fh;
              int64_t w = ow * stride - padding + fw;
              if (h < 0 || h >= dataH || w < 0 || w >= dataW) continue;
              sum += data[0][c][h][w] * filter[f][c][fh][fw];
            }
          }
        }
        result[0][f][oh][ow] = sum;
      }
    }
  }
  return result;
}

std::vector<std::vector<int>> reference2dConvChwFchwMatrix(
    const ConvTensor4D& filter, int64_t dataH, int64_t dataW, int64_t stride,
    int64_t padding) {
  int64_t outputChannels = filter.size();
  int64_t inputChannels = filter[0].size();
  int64_t filterH = filter[0][0].size();
  int64_t filterW = filter[0][0][0].size();
  int64_t outputH = convOutputExtent(dataH, filterH, stride, padding);
  int64_t outputW = convOutputExtent(dataW, filterW, stride, padding);

  std::vector<std::vector<int>> matrix(
      outputChannels * outputH * outputW,
      std::vector<int>(inputChannels * dataH * dataW, 0));
  for (int64_t f = 0; f < outputChannels; ++f) {
    for (int64_t oh = 0; oh < outputH; ++oh) {
      for (int64_t ow = 0; ow < outputW; ++ow) {
        for (int64_t c = 0; c < inputChannels; ++c) {
          for (int64_t fh = 0; fh < filterH; ++fh) {
            for (int64_t fw = 0; fw < filterW; ++fw) {
              int64_t h = oh * stride - padding + fh;
              int64_t w = ow * stride - padding + fw;
              if (h < 0 || h >= dataH || w < 0 || w >= dataW) continue;
              matrix[(f * outputH + oh) * outputW + ow]
                    [(c * dataH + h) * dataW + w] = filter[f][c][fh][fw];
            }
          }
        }
      }
    }
  }
  return matrix;
}

FailureOr<presburger::IntegerRelation>
get2dConvChwFchwFilterDiagonalizedRelation(RankedTensorType filterType,
                                           RankedTensorType dataType,
                                           ArrayRef<int64_t> strides,
                                           int64_t padding,
                                           int64_t minSlotCount,
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
        diagonalize2dMatrix(rowInterchangeRelation, filterType, minSlotCount);
    if (failed(diagonalizedInterchange)) return failure();

    expandedFilterRelation.compose(diagonalizedInterchange.value());
    return expandedFilterRelation;
  }
  auto res =
      diagonalize2dMatrix(expandedFilterRelation, filterType, minSlotCount);
  return res;
}

}  // namespace heir
}  // namespace mlir
