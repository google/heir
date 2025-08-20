#include "lib/Utils/Layout/Hoisting.h"

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

// ISL
#include "include/isl/ast.h"             // from @isl
#include "include/isl/ast_build.h"       // from @isl
#include "include/isl/ast_type.h"        // from @isl
#include "include/isl/constraint.h"      // from @isl
#include "include/isl/ctx.h"             // from @isl
#include "include/isl/local_space.h"     // from @isl
#include "include/isl/map.h"             // from @isl
#include "include/isl/map_type.h"        // from @isl
#include "include/isl/set.h"             // from @isl
#include "include/isl/space.h"           // from @isl
#include "include/isl/space_type.h"      // from @isl
#include "include/isl/union_map.h"       // from @isl
#include "include/isl/union_map_type.h"  // from @isl

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

}  // namespace heir
}  // namespace mlir
