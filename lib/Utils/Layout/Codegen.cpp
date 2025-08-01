#include "lib/Utils/Layout/Codegen.h"

#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"  // from @llvm-project

#define DEBUG_TYPE "layout-codegen"

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;

FailureOr<std::pair<int64_t, int64_t>> getProjectedBounds(
    const IntegerRelation& rel, int varIndex) {
  std::unique_ptr<IntegerRelation> cloneRel = rel.clone();
  cloneRel->projectOut(0, varIndex);
  cloneRel->projectOut(1, cloneRel->getNumVars() - 1);
  std::optional<int64_t> lower = cloneRel->getConstantBound64(BoundType::LB, 0);
  std::optional<int64_t> upper = cloneRel->getConstantBound64(BoundType::UB, 0);

  if (!lower.has_value() || !upper.has_value()) {
    return failure();
  }
  return std::make_pair(lower.value(), upper.value());
}

DenseMap<unsigned, AffineExpr> identifyLocalExprs(const IntegerRelation& rel,
                                                  MLIRContext* context) {
  DenseMap<unsigned, AffineExpr> localExprs;
  if (rel.getNumLocalVars() == 0) {
    return localExprs;
  }
  localExprs.reserve(rel.getNumLocalVars());

  // index i corresponds to the i-th variable in the relation.
  SmallVector<AffineExpr, 4> dims;
  for (unsigned i = 0; i < rel.getNumVars(); ++i) {
    dims.push_back(getAffineDimExpr(i, context));
  }

  // For each local variable, we find an equality that uses the variable and
  // isolate it.
  for (unsigned j = rel.getVarKindOffset(VarKind::Local);
       j < rel.getVarKindEnd(VarKind::Local); ++j) {
    // This always finds the first equality with a non-zero coefficient, though
    // perhaps we may need to find a different one, if the local variable we're
    // eliminating has a simpler equality that shows up later, e.g., maybe the
    // first equality found has another local variable in it, but a later one
    // has only domain/range vars.
    //
    // FIXME: replace this with a search for the first equality that has only
    // domain/range variables in it, or local variables that have already been
    // eliminated.
    std::optional<unsigned> nonzeroConstraintIndex =
        rel.findConstraintWithNonZeroAt(j, /*isEq=*/true);
    if (!nonzeroConstraintIndex.has_value()) {
      localExprs[j] = getAffineConstantExpr(0, context);  // local var == zero
    }
    SmallVector<int64_t> constraint =
        rel.getEquality64(nonzeroConstraintIndex.value());

    // The coefficient of the local variable is divided through the rest of the
    // expr. We negate the coefficient of the local variable because we move it
    // to the other side of the equality before dividing. E.g.,
    //
    // Tableau row 1  -1  -2  0  == 0 corresponds to d0 - d1 - 2q + 0 == 0
    //
    // And to isolate q we convert it to
    //
    //   d0 - d1 == 2q
    //
    // and divide through by 2.
    int64_t divisor = -constraint[j];
    AffineExpr localExpr = getAffineConstantExpr(0, context);
    for (unsigned i = 0; i < constraint.size(); ++i) {
      if (i == j) continue;              // Don't include the local variable.
      if (constraint[i] == 0) continue;  // Skip zero coefficients.
      localExpr = localExpr + dims[i] * constraint[i];
    }
    localExpr = localExpr.floorDiv(divisor);

    // LLVM_DEBUG(
    llvm::errs() << "Identified local expr for variable " << j << ": "
                 << localExpr << "\n";
    // );

    localExprs[j] = localExpr;
  }

  return localExprs;
}

FailureOr<LoopNest> generateLoopNest(const IntegerRelation& rel,
                                     MLIRContext* context) {
  LoopNest nest;
  std::unique_ptr<IntegerRelation> cloneRel = rel.clone();
  cloneRel->dump();
  cloneRel->simplify();
  cloneRel->removeTrivialRedundancy();
  cloneRel->dump();
  SmallVector<AffineDimExpr, 4> inductionVars;

  // This vector stores local variables that can be represented in terms of
  // other variables in the relation, so that they can be substituted out when
  // reconstructing the affine exprs.
  DenseMap<unsigned, AffineExpr> localExprs =
      identifyLocalExprs(*cloneRel, context);

  // FIXME: Figure out what to do if only some local variables can be
  // eliminated. (Iterate over them, probably)
  assert(localExprs.size() == cloneRel->getNumLocalVars() &&
         "Expected all local variables to be eliminated");
  SmallVector<AffineExpr, 4> localExprsVec(localExprs.values());

  // Induction vars are only those which can't be eliminated.
  // FIXME: support eliminiating domain/range vars when possible.
  nest.numInductionVars = 0;
  for (int i = 0; i < cloneRel->getVarKindEnd(VarKind::Range); ++i) {
    auto res = getProjectedBounds(*cloneRel, i);
    if (failed(res)) {
      return failure();
    }
    auto [lower, upper] = res.value();
    nest.lowerBounds.push_back(lower);
    nest.upperBounds.push_back(upper);
    nest.numInductionVars++;
  }

  for (int i = 0; i < cloneRel->getNumEqualities(); ++i) {
    SmallVector<int64_t> constraint = cloneRel->getEquality64(i);
    AffineExpr expr = getAffineExprFromFlatForm(
        constraint, nest.numInductionVars, 0, localExprsVec, context);
    nest.constraints.push_back(expr);
    nest.eq.push_back(true);
  }

  for (int i = 0; i < cloneRel->getNumInequalities(); ++i) {
    SmallVector<int64_t> constraint = cloneRel->getInequality64(i);
    AffineExpr expr = getAffineExprFromFlatForm(
        constraint, nest.numInductionVars, 0, localExprsVec, context);
    nest.constraints.push_back(expr);
    nest.eq.push_back(false);
  }
  return nest;
}

}  // namespace heir
}  // namespace mlir
