#include "lib/Utils/Layout/Hoisting.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

// ISL
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

using llvm::SmallVector;
using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

presburger::IntegerRelation hoistConversionThroughMatvec(
    const IntegerRelation& matrixLayout, const IntegerRelation& fromVecLayout,
    const IntegerRelation& toVecLayout) {
  // The intuition for this function is that the conversion from fromVecLayout
  // to toVecLayout implies some transformation of the slot ordering of the
  // packed vector. The kernels we support have matrix layouts for which the
  // packed slots of a ciphertext track the columns of the packed vector.
  // So we need to apply the same transformation of the packed vector slots
  // to the packed matrix slots.
  //
  // This function works in two steps:
  //
  // 1. Compute the inferred re-packing relation of vector slots (i.e., a
  // relation (ct, slots) -> (ct, slots))
  //
  // 2. Compose (1) with the (ct, slot) dims of the matrix packing.

  IntegerRelation fromClone(fromVecLayout);
  IntegerRelation toClone(toVecLayout);

  // Project out the ciphertext dimension, though this will need to change when
  // we get a larger vector than can fit in one ciphertext. This is where the
  // assumption about the vector packing fitting into one ciphertext comes from.
  assert(fromClone.getConstantBoundOnDimSize64(1).value() == 1);
  assert(toClone.getConstantBoundOnDimSize64(1).value() == 1);
  fromClone.projectOut(1);
  toClone.projectOut(1);

  fromClone.inverse();
  fromClone.compose(toClone);
  fromClone.removeRedundantConstraints();
  fromClone.simplify();

  // At this stage, fromClone models the re-packing relation on the vector
  // (slots) -> (slots). Put the ct dim back in with bounds from the matrix
  // layout, and equality between the domain ct var and the range ct var.
  fromClone.insertVar(VarKind::Domain, 0, 1);
  fromClone.insertVar(VarKind::Range, 0, 1);
  std::optional<int64_t> ctUb = matrixLayout.getConstantBound64(
      BoundType::UB, matrixLayout.getVarKindOffset(VarKind::Range));
  std::optional<int64_t> ctLb = matrixLayout.getConstantBound64(
      BoundType::LB, matrixLayout.getVarKindOffset(VarKind::Range));

  if (ctUb.has_value()) {
    fromClone.addBound(BoundType::UB,
                       fromClone.getVarKindOffset(VarKind::Domain),
                       ctUb.value());
    fromClone.addBound(BoundType::UB,
                       fromClone.getVarKindOffset(VarKind::Range),
                       ctUb.value());
  }
  if (ctLb.has_value()) {
    fromClone.addBound(BoundType::LB,
                       fromClone.getVarKindOffset(VarKind::Domain),
                       ctLb.value());
    fromClone.addBound(BoundType::LB,
                       fromClone.getVarKindOffset(VarKind::Range),
                       ctLb.value());
  }

  // Still need to ensure the input and output ciphertext
  // are the same (i.e., slots can only be rotated within one ct).
  // Order of variables in the constraint are:
  //
  //    0,    1,  2,    3,        4
  //   ct, slot, ct, slot, constant
  //
  SmallVector<int64_t> ciphertextEq(fromClone.getNumCols(), 0);
  ciphertextEq[0] = 1;
  ciphertextEq[2] = -1;
  fromClone.addEquality(ciphertextEq);

  // At this point the re-packing relation should look something like
  //
  // {
  //   [i0, i1] -> [o0, o1] :     (ct, slot) -> (ct, slot)
  //   (3 - i1 + o1) mod 8 = 0    slot transformation from vec conversion
  //   and i0 = o0                ct equality constraint
  //   and 0 <= i0 <= 8           bounds on ct dim from matrixLayout
  //   and 0 <= o0 <= 8
  //   and 0 <= i1 <= 15          bounds on slot dim
  //   and 0 <= o1 <= 15
  // }
  //
  IntegerRelation result(matrixLayout);
  result.compose(fromClone);
  return result;
}

FailureOr<presburger::IntegerRelation> pushSliceLayoutThroughInsertSlice(
    SmallVector<int64_t> insertSliceSizes, ArrayRef<int64_t> resultShape,
    const presburger::IntegerRelation& sliceLayout) {
  // Check if the slice we insert fills up a full slice of the output tensor.
  SmallVector<int64_t> unitDims;
  for (auto [idx, size] : llvm::enumerate(insertSliceSizes)) {
    if (size != 1) {
      // Non-unit sizes must equal the input slice.
      assert(size == resultShape[idx]);
      continue;
    }
    // We have a unit dimension. If the unit dimension matches with an output
    // size 1 dimension, then we can drop the dimension. Otherwise, we are
    // inserting into a single slice of a larger dimension, so collect the
    // dimension for later.
    unitDims.push_back(idx);
  }

  // Get the number of ct that each slice takes up.
  auto numCt = sliceLayout.getConstantBound64(
      BoundType::UB, sliceLayout.getVarKindOffset(VarKind::Range));
  if (!numCt.has_value()) {
    return failure();
  }
  auto numSlots = sliceLayout.getConstantBound64(
      BoundType::UB, sliceLayout.getVarKindOffset(VarKind::Range) + 1);
  if (!numSlots.has_value()) {
    return failure();
  }

  // Now the slice takes up a full subtensor of the output tensor. For the
  // result layout, let this subtensor be preserved.
  auto resultLayout = sliceLayout.clone();
  // Add domain variables in the positions of the unit dimensions.
  for (int i = 0; i < resultShape.size(); ++i) {
    if (llvm::is_contained(unitDims, i)) {
      auto domainVar = resultLayout->insertVar(VarKind::Domain, i, 1);
      addBounds(*resultLayout, domainVar, 0, resultShape[i] - 1);
    }
  }
  // Add a range var to indicate the index of the slice.
  auto newRangeVar = resultLayout->insertVar(VarKind::Range, 0, 1);
  addBounds(*resultLayout, newRangeVar, 0);

  // Create a relation mapping the new unit dimensions to a new range variable r
  // that represents the index into each (ct, slot) per slice.
  SmallVector<int64_t> newRangeCoeffs(resultLayout->getNumCols(), 0);
  unsigned product = 1;
  for (auto dim : unitDims) {
    newRangeCoeffs[dim] = product;
    product *= resultShape[dim];
  }
  newRangeCoeffs[newRangeVar] = -1;
  resultLayout->addEquality(newRangeCoeffs);

  // Now compose the relation with a new relation mapping (r, ct, slot) -> (r *
  // num_ct + ct, slot).
  IntegerRelation newRangeRelation(PresburgerSpace::getRelationSpace(
      /*numDomain=*/3, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/0));
  auto domainOffset = newRangeRelation.getVarKindOffset(VarKind::Domain);
  auto rangeOffset = newRangeRelation.getVarKindOffset(VarKind::Range);
  addBounds(newRangeRelation, domainOffset, 0);
  addBounds(newRangeRelation, domainOffset + 1, 0, numCt.value());
  addBounds(newRangeRelation, domainOffset + 2, 0, numSlots.value());
  addBounds(newRangeRelation, rangeOffset + 1, 0, numSlots.value());
  // slot = slot
  addConstraint(newRangeRelation,
                {{rangeOffset + 1, 1}, {domainOffset + 2, -1}},
                /*equality=*/true);
  // r * num_ct + ct = ct'
  addConstraint(newRangeRelation,
                {{rangeOffset, -1},
                 {domainOffset, numCt.value() + 1},
                 {domainOffset + 1, 1}},
                /*equality=*/true);
  resultLayout->compose(newRangeRelation);

  return *resultLayout;
}

}  // namespace heir
}  // namespace mlir
