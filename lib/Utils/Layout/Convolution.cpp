#include "lib/Utils/Layout/Convolution.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

presburger::IntegerRelation get2dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding) {
  auto domainSize = filterType.getRank();
  assert(domainSize == 2 && "expected 2-D filter matrix");

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      domainSize, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/2));

  // Filter row and column indices
  auto filterRow = result.getVarKindOffset(VarKind::Domain);
  auto filterCol = result.getVarKindOffset(VarKind::Domain) + 1;

  // Matrix row and column indices
  auto matRow = result.getVarKindOffset(VarKind::Range);
  auto matCol = result.getVarKindOffset(VarKind::Range) + 1;

  // Constant coefficient
  auto constCoeff = result.getNumCols() - 1;

  // Filter, datasize, and strides.
  auto filterRowSize = filterType.getDimSize(0);
  auto filterColSize = filterType.getDimSize(1);
  auto dataRowSize = dataType.getDimSize(0);
  auto dataColSize = dataType.getDimSize(1);
  auto strideRow = strides[0];
  auto strideCol = strides[1];

  // These are the indices that represent the valid positions that the filter
  // can move over the data. (0, 0) is the first position of (slidingRow,
  // slidingCol).
  auto slidingRow = result.getVarKindOffset(VarKind::Local);
  auto slidingCol = result.getVarKindOffset(VarKind::Local) + 1;

  // The maximum values for the sliding window indices.
  auto slidingRowSize =
      (dataRowSize + 2 * padding - filterRowSize) / strideRow + 1;
  auto slidingColSize =
      (dataColSize + 2 * padding - filterColSize) / strideCol + 1;

  // Add bounds for the filter matrix dimensions.
  addBounds(result, filterRow, 0, filterRowSize - 1);
  addBounds(result, filterCol, 0, filterColSize - 1);

  // Add bounds for the sliding window indices.
  addBounds(result, slidingRow, 0, slidingRowSize - 1);
  addBounds(result, slidingCol, 0, slidingColSize - 1);

  // Define (dataRow, dataCol) to be the position on the data tensor for a given
  // filter position (slidingRow, slidingCol) and a given filter index
  // (filterRow, filterCol). E.g. the top left corner of the filter is at
  // (filterRow, filterCol) = (0, 0) and the first position of the filter is at
  // (slidingRow, slidingCol) = (0, 0). This corresponds to (-padding, -padding)
  // on the data indices (dataRow, dataCol).
  //   dataRow = (slidingRow * strideRow - padding) + filterRow
  //   dataCol = (slidingCol * strideCol - padding) + filterCol

  // Add constraints for when the filter sliding window index is at a valid
  // data position. Require:
  //    0 <= dataRow < dataRowSize and 0 <= dataCol < dataColSize.
  //  Substituting the expressions gives:
  //    0 <= slidingRow * strideRow - padding + filterRow < dataRowSize
  addConstraint(
      result, {{slidingRow, strideRow}, {filterRow, 1}, {constCoeff, -padding}},
      /*equality=*/false);
  addConstraint(result,
                {{constCoeff, dataRowSize + padding - 1},
                 {slidingRow, -strideRow},
                 {filterRow, -1}},
                /*equality=*/false);

  // 0 <= slidingCol * strideCol - padding + filterCol < dataColSize
  addConstraint(
      result, {{slidingCol, strideCol}, {filterCol, 1}, {constCoeff, -padding}},
      /*equality=*/false);
  addConstraint(result,
                {{constCoeff, dataColSize + padding - 1},
                 {slidingCol, -strideCol},
                 {filterCol, -1}},
                /*equality=*/false);

  // Add equalities for the resulting matrix row and column. Each matrix row
  // corresponds to one sliding window of the filter over the data. So flatten
  // the filter sliding window indices (slidingRow, slidingCol):
  // matRow = slidingRow * slidingColSize + slidingCol
  addConstraint(result,
                {{matRow, -1}, {slidingRow, slidingColSize}, {slidingCol, 1}},
                /*equality=*/true);

  // The matrix column is the flattened data indices:
  // matCol = dataRow * dataColSize + dataCol
  // matCol = (slidingRow * strideRow - padding + filterRow) * dataColSize +
  //          (slidingCol * strideCol - padding + filterCol)
  // matCol = slidingRow * strideRow * dataColSize - padding * dataColSize +
  //          filterRow * dataColSize + slidingCol * strideCol - padding +
  //          filterCol
  addConstraint(result,
                {{matCol, -1},
                 {slidingRow, strideRow * dataColSize},
                 {slidingCol, strideCol},
                 {filterRow, dataColSize},
                 {filterCol, 1},
                 {constCoeff, -padding * dataColSize - padding}},
                /*equality=*/true);
  return result;
}

presburger::IntegerRelation matrixRowStridingRelation(int64_t rowSize,
                                                      int64_t colSize,
                                                      int64_t stride) {
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      2, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/1));

  // Initial row and column indices
  auto matRow = result.getVarKindOffset(VarKind::Domain);
  auto matCol = result.getVarKindOffset(VarKind::Domain) + 1;

  auto resultRow = result.getVarKindOffset(VarKind::Range);
  auto resultCol = result.getVarKindOffset(VarKind::Range) + 1;

  addBounds(result, matRow, 0, rowSize);
  addBounds(result, matCol, 0, colSize);

  // Only pick rows whose index is divisible by stride:
  // matRow = stride *resultRow
  addConstraint(result, {{matRow, 1}, {resultRow, -stride}}, true);
  // Keep all the columns
  addConstraint(result, {{matCol, 1}, {resultCol, -1}}, true);
  return result;
}

// We iterate over i the position of the start of the filter over the data
// (starting from -padding and with 0 indexing over the data)
//
// -padding <= i <= dataSize - filterSize + padding
//
// matRow = padding + i (hence matRow <=  dataSize - kernelSize + 2*padding)
//
// matCol = i + filterIndex
presburger::IntegerRelation get1dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    int64_t stride,
                                                    int64_t padding) {
  auto domainSize = filterType.getRank();
  assert(domainSize == 1 && "expected 1-D filter matrix");

  IntegerRelation result(PresburgerSpace::getRelationSpace(
      domainSize, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/1));

  auto filterIndex = result.getVarKindOffset(VarKind::Domain);

  // Matrix row and column indices
  auto matRow = result.getVarKindOffset(VarKind::Range);
  auto matCol = result.getVarKindOffset(VarKind::Range) + 1;

  auto i = result.getVarKindOffset(VarKind::Local);

  // Constant coefficient
  auto constCoeff = result.getNumCols() - 1;

  // Filter, datasize
  auto filterSize = filterType.getDimSize(0);
  auto dataSize = dataType.getDimSize(0);

  // The maximum values for the sliding window indices.
  auto slidingSize = dataSize + padding - filterSize;

  // Add bounds for the filter matrix dimensions.
  addBounds(result, filterIndex, 0, filterSize - 1);

  // Add bounds for the sliding window indices.
  addBounds(result, i, -padding, slidingSize);

  // Add bound for the column index
  addBounds(result, matCol, 0, dataSize - 1);

  addBounds(result, matRow, 0, slidingSize + padding);

  // Note that i starts at -padding
  // matRow = i + padding
  addConstraint(result, {{matRow, 1}, {i, -1}, {constCoeff, -padding}}, true);
  // matCol = i + filterIndex
  addConstraint(
      result, {{matCol, 1}, {i, -1}, {filterIndex, -1}, {constCoeff, 0}}, true);

  if (stride > 1) {
    auto rowSelector =
        matrixRowStridingRelation(slidingSize + padding, dataSize - 1, stride);
    result.applyRange(rowSelector);
  }

  return result;
}

RankedTensorType get1dConvFilterExpandedType(RankedTensorType filterType,
                                             RankedTensorType dataType,
                                             int64_t stride, int64_t padding) {
  auto filterSize = filterType.getDimSize(0);
  auto dataSize = dataType.getDimSize(0);

  int64_t rows = (dataSize + 2 * padding - filterSize) / stride + 1;

  // Number of columns will be the number of data elements.
  int64_t cols = dataSize;

  return RankedTensorType::get({rows, cols}, filterType.getElementType());
}

RankedTensorType get2dConvFilterExpandedType(RankedTensorType filterType,
                                             RankedTensorType dataType,
                                             int64_t padding,
                                             ArrayRef<int64_t> strides) {
  auto filterRowSize = filterType.getDimSize(0);
  auto filterColSize = filterType.getDimSize(1);
  auto dataRowSize = dataType.getDimSize(0);
  auto dataColSize = dataType.getDimSize(1);
  auto strideRow = strides[0];
  auto strideCol = strides[1];

  // Number of rows will be the filter sliding rows * filter sliding columns.
  int64_t filterSlidingRows =
      (dataRowSize + 2 * padding - filterRowSize) / strideRow + 1;
  int64_t filterSlidingCols =
      (dataColSize + 2 * padding - filterColSize) / strideCol + 1;
  int64_t rows = filterSlidingRows * filterSlidingCols;

  // Number of columns will be the number of data elements.
  int64_t cols = dataType.getNumElements();

  return RankedTensorType::get({rows, cols}, filterType.getElementType());
}

RankedTensorType get2dConvChwFchwFilterExpandedType(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    int64_t padding,
                                                    ArrayRef<int64_t> strides) {
  // Get the filter relation for a single input and output channel and multiply
  // the dimensions by the number of input and output channels for the row and
  // column dimensions respectively.
  RankedTensorType singleFilterType = RankedTensorType::get(
      {filterType.getDimSize(2), filterType.getDimSize(3)},
      filterType.getElementType());
  RankedTensorType singleDataType =
      RankedTensorType::get({dataType.getDimSize(2), dataType.getDimSize(3)},
                            dataType.getElementType());
  auto singleResultType = get2dConvFilterExpandedType(
      singleFilterType, singleDataType, padding, strides);

  int64_t inputChannels = dataType.getDimSize(1);
  int64_t outputChannels = filterType.getDimSize(0);

  int64_t rows = outputChannels * singleResultType.getDimSize(0);
  int64_t cols = inputChannels * singleResultType.getDimSize(1);
  return RankedTensorType::get({rows, cols}, filterType.getElementType());
}

presburger::IntegerRelation get2dConvChwFchwFilterRelation(
    RankedTensorType filterType, RankedTensorType dataType,
    ArrayRef<int64_t> strides, int64_t padding) {
  assert(filterType.getRank() == 4 && "expected 4-D filter matrix");
  assert(dataType.getRank() == 4 && "expected 4-D data matrix");
  assert(dataType.getDimSize(0) == 1 && "expected N=1 batch size");

  // Get the filter relation for a single input and output channel.
  RankedTensorType singleFilterType = RankedTensorType::get(
      {filterType.getDimSize(2), filterType.getDimSize(3)},
      filterType.getElementType());
  RankedTensorType singleDataType =
      RankedTensorType::get({dataType.getDimSize(2), dataType.getDimSize(3)},
                            dataType.getElementType());
  auto singleFilterRelation = get2dConvFilterRelation(
      singleFilterType, singleDataType, strides, padding);

  // Map the single filter relation into the multi-channel matrix. Each single
  // filter is offset into the result by adding (c * totalRowSize, f *
  // totalColSize) to the range dimensions.

  // First, add (f, c) to the domain vars and set bounds
  singleFilterRelation.insertVar(presburger::VarKind::Domain, 0, 2);
  auto fDim =
      singleFilterRelation.getVarKindOffset(presburger::VarKind::Domain);
  auto cDim =
      singleFilterRelation.getVarKindOffset(presburger::VarKind::Domain) + 1;

  auto inputChannels = dataType.getDimSize(1);
  auto outputChannels = filterType.getDimSize(0);
  assert(inputChannels == filterType.getDimSize(1) &&
         "input channels must match filter input channels");
  addBounds(singleFilterRelation, fDim, 0, outputChannels - 1);
  addBounds(singleFilterRelation, cDim, 0, inputChannels - 1);

  // Expand the range vars so that we can compose with the embedding relation.
  singleFilterRelation.insertVar(presburger::VarKind::Range, 0, 2);
  // (embedRow, embedCol) = position in the embedded matrix.
  // (singleRow, singleCol) = position in the single filter matrix.
  auto embedRow =
      singleFilterRelation.getVarKindOffset(presburger::VarKind::Range);
  auto embedCol =
      singleFilterRelation.getVarKindOffset(presburger::VarKind::Range) + 1;
  auto singleRow =
      singleFilterRelation.getVarKindOffset(presburger::VarKind::Range) + 2;
  auto singleCol =
      singleFilterRelation.getVarKindOffset(presburger::VarKind::Range) + 3;

  // embedRow = fDim * totalRowSize +  singleRow
  // embedCol = cDim * totalColSize + singleCol
  auto singleResultType = get2dConvFilterExpandedType(
      singleFilterType, singleDataType, padding, strides);
  auto totalRowSize = singleResultType.getDimSize(0);
  auto totalColSize = singleResultType.getDimSize(1);

  addConstraint(singleFilterRelation,
                {{embedRow, 1}, {fDim, -totalRowSize}, {singleRow, -1}}, true);
  addConstraint(singleFilterRelation,
                {{embedCol, 1}, {cDim, -totalColSize}, {singleCol, -1}}, true);

  // Add bounds for the matrix dimensions.
  addBounds(singleFilterRelation, embedRow, 0,
            outputChannels * totalRowSize - 1);
  addBounds(singleFilterRelation, embedCol, 0,
            inputChannels * totalColSize - 1);

  // Project out the single filter relation range vars.
  singleFilterRelation.projectOut(singleRow, 2);

  return singleFilterRelation;
}

FailureOr<presburger::IntegerRelation> getConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize) {
  if (filterType.getRank() == 1) {
    int64_t stride = 1;
    auto filterRelation =
        get1dConvFilterRelation(filterType, dataType, stride, padding);
    return diagonalize2dMatrix(filterRelation, filterType, ciphertextSize);
  }
  if (filterType.getRank() != 2) return failure();
  SmallVector<int64_t> strides = {1, 1};
  auto filterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);
  return diagonalize2dMatrix(filterRelation, filterType, ciphertextSize);
}

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

    auto rowInterchangeRelation = getRowInterchangeRelation(
        filterType.getDimSize(0), outputH, outputW, strides[0]);
    rowInterchangeRelation.appendVar(presburger::VarKind::Domain);
    rowInterchangeRelation.appendVar(presburger::VarKind::Range);
    addConstraint(
        rowInterchangeRelation,
        {{rowInterchangeRelation.getVarKindOffset(presburger::VarKind::Domain) +
              1,
          -1},
         {rowInterchangeRelation.getVarKindOffset(presburger::VarKind::Range) +
              1,
          1}},
        /*equality=*/true);
    expandedFilterRelation.compose(rowInterchangeRelation);
  }
  return diagonalize2dMatrix(expandedFilterRelation, filterType,
                             ciphertextSize);
}

bool isRelationConvFilterDiagonalized(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize, const presburger::IntegerRelation& relation) {
  auto diagonalizedRelation = getConvFilterDiagonalizedRelation(
      filterType, dataType, padding, ciphertextSize);
  if (failed(diagonalizedRelation)) {
    return false;
  }
  return isRelationEqual(relation, diagonalizedRelation.value());
}

presburger::IntegerRelation getRowInterchangeRelation(int64_t c, int64_t h,
                                                      int64_t w, int64_t g) {
  // 1. Unflatten idx_in (H, W, C*g^2) into (hi, wi, ci).
  // 2. Map to output coordinates by interleaving sub-pixels:
  //    h' = hi * g + (ci % g**2) // g
  //    w' = wi * g + (ci % g)
  // 3. Flatten (gW, gH, C) into idx_out = (c * g * h) * w' + (c) * h' + c'
  int64_t hOut = w * g;
  int64_t wOut = h * g;
  int64_t cOut = c / (g * g);
  int64_t numElements = c * h * w;

  // One to one mapping from idx_in to idx_out.
  std::string islStr = llvm::formatv(
      "{{ [idx_in] -> [idx_out] : exists hi, wi, ci, ho, wo, co : "
      "0 <= hi < {0} and 0 <= wi < {1} and 0 <= ci < {2} and "
      "0 <= ho < {3} and 0 <= wo < {4} and 0 <= co < {5} and co = ci // "
      "{10}^2 and "
      "wo = wi * {10} + (ci % {10}) and "
      "ho = hi * {10} + (ci % {10}^2) // {10} and "
      "idx_in = wi + hi * {6} + ci * {7} and "
      "idx_out = wo + ho * {8} + co * {9} and 0 <= idx_in < {11} and 0 <= "
      "idx_out < {11} }",
      h, w, c, hOut, wOut, cOut, w, h * w, wOut, hOut * wOut, g, numElements);

  return getIntegerRelationFromIslStr(islStr).value();
}

presburger::IntegerRelation get2dConvResultRelation(RankedTensorType outputType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding,
                                                    int64_t ciphertextSize,
                                                    bool interchangeRows) {
  assert(llvm::all_equal(strides) && "strides must be equal");

  // First flatten the output tensor into a 1-D tensor of (ct, slot) where ct =
  // 0 (set the "ciphertextSize" to be the same as the number of elements). This
  // creates outputType -> [0, slot].
  auto flattenedOutput =
      getRowMajorLayoutRelation(outputType, outputType.getNumElements());

  int64_t numCiphertexts =
      std::ceil((float)outputType.getNumElements() / ciphertextSize);
  int64_t paddedSize = isPowerOfTwo(outputType.getNumElements())
                           ? outputType.getNumElements()
                           : nextPowerOfTwo(outputType.getNumElements());

  // Create the interchange permutation [idx_in] -> [idx_out] and add a domain
  // var = 0 to align with the range of the flattenedOutput relation.
  if (interchangeRows) {
    int64_t c = outputType.getDimSize(1);
    int64_t h = outputType.getDimSize(2);
    int64_t w = outputType.getDimSize(3);
    int64_t g = strides[0];
    auto rowInterchange = getRowInterchangeRelation(c, h, w, g);
    rowInterchange.insertVar(presburger::VarKind::Domain, 0);
    addConstraint(
        rowInterchange,
        {{rowInterchange.getVarKindOffset(presburger::VarKind::Domain), 1},
         {rowInterchange.getNumCols() - 1, 0}},
        /*equality=*/true);
    // Compose the row interchange relation with the flattened output relation:
    // [outputType] -> [0, slot] -> [slot'].
    flattenedOutput.compose(rowInterchange);

    // Compose with the [slot'] -> [ct, slot] relation across multiple
    // ciphertexts.
    std::string mapToCtSlot = llvm::formatv(
        "{{ [idx_out] -> [ct, slot] : "
        "0 <= ct < {0} and 0 <= slot < {2} and slot % {1} = idx_out % {2} and "
        "ct = idx_out // {2} }",
        numCiphertexts, paddedSize, ciphertextSize);
    auto toCtSlot = getIntegerRelationFromIslStr(mapToCtSlot).value();
    flattenedOutput.compose(toCtSlot);
    return flattenedOutput;
  }

  std::string mapToCtSlot = llvm::formatv(
      "{{ [in_ct, idx_out] -> [ct, slot] : in_ct = 0 and "
      "0 <= ct < {0} and 0 <= slot < {2} and slot % {1} = idx_out % {2} and "
      "ct = idx_out // {2} }",
      numCiphertexts, paddedSize, ciphertextSize);
  auto toCtSlot = getIntegerRelationFromIslStr(mapToCtSlot).value();
  flattenedOutput.compose(toCtSlot);
  return flattenedOutput;

  return flattenedOutput;
}

}  // namespace heir
}  // namespace mlir
