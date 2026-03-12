#include "lib/Utils/Layout/Convolution.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
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

presburger::IntegerRelation get2dConvChwFchwFilterRelation(
    RankedTensorType filterType, RankedTensorType dataType,
    ArrayRef<int64_t> strides, int64_t padding) {
  assert(filterType.getRank() == 4 && "expected 4-D filter matrix");
  assert(dataType.getRank() == 3 && "expected 3-D data matrix");

  // Get the filter relation for a single input and output channel.
  RankedTensorType singleFilterType = RankedTensorType::get(
      {filterType.getDimSize(2), filterType.getDimSize(3)},
      filterType.getElementType());
  RankedTensorType singleDataType =
      RankedTensorType::get({dataType.getDimSize(1), dataType.getDimSize(2)},
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

  auto inputChannels = dataType.getDimSize(0);
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
  // Project out the single filter relation range vars.
  singleFilterRelation.projectOut(singleRow, 2);

  return singleFilterRelation;
}

FailureOr<presburger::IntegerRelation> get2dConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize) {
  SmallVector<int64_t> strides = {1, 1};
  auto expandedFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);
  // Get size of the expanded filter matrix.
  auto rowBound = expandedFilterRelation.getConstantBound64(
      BoundType::UB, expandedFilterRelation.getVarKindOffset(VarKind::Range));
  if (!rowBound.has_value()) {
    return failure();
  }
  auto colBound = expandedFilterRelation.getConstantBound64(
      BoundType::UB,
      expandedFilterRelation.getVarKindOffset(VarKind::Range) + 1);
  if (!colBound.has_value()) {
    return failure();
  }
  RankedTensorType expandedFilterType =
      RankedTensorType::get({rowBound.value() + 1, colBound.value() + 1},
                            filterType.getElementType());

  auto diagonalizedFilterRelation =
      getDiagonalLayoutRelation(expandedFilterType, ciphertextSize);

  // Compose these relations.
  expandedFilterRelation.compose(diagonalizedFilterRelation);
  return expandedFilterRelation;
}

bool isRelation2dConvFilterDiagonalized(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t ciphertextSize, const presburger::IntegerRelation& relation) {
  auto diagonalizedRelation = get2dConvFilterDiagonalizedRelation(
      filterType, dataType, padding, ciphertextSize);
  if (failed(diagonalizedRelation)) {
    return false;
  }
  bool fastCheck = relation.isObviouslyEqual(diagonalizedRelation.value());
  if (fastCheck) return true;

  LogicalResult inequalityTest =
      tryProveUnequal(diagonalizedRelation.value(), relation);
  if (succeeded(inequalityTest)) return false;

  bool slowCheck = relation.isEqual(diagonalizedRelation.value());
  return slowCheck;
}

presburger::IntegerRelation getRowInterchangeRelation(int64_t c, int64_t h,
                                                      int64_t w, int64_t g) {
  // Map flattened indices from a channel-last (H, W, C*g^2) tensor to a
  // channel-first (C, gH, gW) tensor using a pixel-shuffle
  // rearrangement. c is the number of semantic channels, h and w are the height
  // and width of the input, and g is the gap size. To interleave the channels,
  // note that each (H, W) 2-D input spans across g*g positions (in the notation
  // below, this means ki is in [0, g^2 - 1]). In the interleaved output, we map
  // ki to an input channel cin by cin = ki mod c. For example if there are 2
  // input channels and gap = 2, then the input channels look like:
  //  * $ki=0$: Channel 0, Sub-pixel $(0,0)$
  //  * $ki=1$: Channel 1, Sub-pixel $(0,0)$
  //  * $ki=2$: Channel 0, Sub-pixel $(0,1)$
  //  * $ki=3$: Channel 1, Sub-pixel $(0,1)$
  //  * ... then repeat ki for 0, 1, 2, 3.
  // This mapping computes which output channel each input pixel maps to. To
  // compute the final coordinates, we have to interleave the pixels within the
  // channel. Within the g^2 sub-pixels of each channel, map the flattened
  // input position p = ki // c into a (g, g) sized position (r, s) with
  // p = r*g + s. Then we map those sub-pixel coordinates (r, s) to the output
  // coordinates with (h' = r * h + hi, w' = s * w + wi). This "expands" each
  // original (H, W) output into the pixel interleaved (gH, gW) output.
  //
  // 1. Unflatten idx_in (H, W, C*g^2) into (hi, wi, ki).
  // 2. Interleave the channels: Map gapped channel index ki into output channel
  // cin and block offsets (r, s), where ki = (r*g + s)*c + cin.
  // 3. Map to output coordinates by interleaving sub-pixels:
  //    h' = r*h + hi, w' = s*w + wi.
  // 4. Flatten into idx_out (C, gH, gW): idx_out = cin*gHgW + h'*gW + w'.
  int64_t totalChannels = c * g * g;
  int64_t hOut = h * g;
  int64_t wOut = w * g;

  // Domain: [idx_in]
  // Range: [ct, slot] where ct=0 and slot=idx_out
  std::string islStr = llvm::formatv(
      "{{ [idx_in] -> [0, idx_out] : exists hi, wi, cin, r, s : "
      "0 <= hi < {0} and 0 <= wi < {1} and 0 <= cin < {2} and "
      "0 <= r < {3} and 0 <= s < {3} and "
      "idx_in = hi * {4} + wi * {5} + (r * {3} + s) * {2} + cin and "
      "idx_out = cin * {6} + (r * {0} + hi) * {7} + (s * {1} + wi) }",
      h, w, c, g, w * totalChannels, totalChannels, hOut * wOut, wOut);

  return getIntegerRelationFromIslStr(islStr).value();
}

}  // namespace heir
}  // namespace mlir
