#ifndef LIB_UTILS_AFFINELOOPUTILS_H_
#define LIB_UTILS_AFFINELOOPUTILS_H_

#include <utility>

#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"    // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

// An affine.if op enclosing some op, together with whether that op is in the
// else region.
using EnclosingAffineIf = std::pair<affine::AffineIfOp, bool>;

// Express `value` as an affine expression whose dimension d_i is `ivs[i]`.
//
// Supports index constants, the given induction variables, and arith
// add/sub/mul (by a constant), floordivsi/remui (by a constant), and
// index_cast ops, as well as affine.apply. `arith.remui` is only interpreted as
// an affine `mod` when the divisor is a power of two, because only then does
// the unsigned remainder of a negative 64-bit index agree with the
// mathematical modulus.
FailureOr<AffineExpr> getAffineExprOfIVs(Value value, ArrayRef<Value> ivs);

// Add a bound of the given type on the variable at `pos` of `cst`, where the
// bound is `ofr` expressed as an affine function of `ivs`. All `ivs` must
// already be variables of `cst`. Upper bounds are exclusive.
LogicalResult addAffineBound(affine::FlatAffineValueConstraints& cst,
                             presburger::BoundType type, unsigned pos,
                             OpFoldResult ofr, ArrayRef<Value> ivs);

// Add the iteration domain of `loop` to `cst`: lb <= iv < ub, and
// iv = lb + step * q for some integer q. The loop must have a single induction
// variable that is a variable of `cst`, and a constant positive step.
LogicalResult addLoopDomain(affine::FlatAffineValueConstraints& cst,
                            LoopLikeOpInterface loop, ArrayRef<Value> ivs);

// Add the constraints of `ifOp`'s integer set (negated if `isElse`) to `cst`.
LogicalResult addAffineIfDomain(affine::FlatAffineValueConstraints& cst,
                                affine::AffineIfOp ifOp, bool isElse,
                                ArrayRef<Value> ivs);

// Collect the loops (innermost first) and affine.if ops enclosing `op`. Fails
// if `op` is nested in an scf.if, whose condition cannot be modeled.
LogicalResult getEnclosingLoopsAndIfs(
    Operation* op, SmallVectorImpl<LoopLikeOpInterface>& loops,
    SmallVectorImpl<EnclosingAffineIf>& affineIfs);

// Build the iteration domain of `loops` (restricted by `affineIfs`) as a set of
// constraints whose dimension variables are the induction variables of
// `loops`, in the given order.
FailureOr<affine::FlatAffineValueConstraints> getIterationDomain(
    ArrayRef<LoopLikeOpInterface> loops,
    ArrayRef<EnclosingAffineIf> affineIfs = {});

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_AFFINELOOPUTILS_H_
