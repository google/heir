#include "lib/Utils/Layout/Codegen.h"

#include <iostream>

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

// Get the bounds of an individual variable by projecting the integer relation
// onto that variable.
FailureOr<std::pair<int64_t, int64_t>> getProjectedBounds(
    const IntegerRelation& rel, int varIndex) {
  std::unique_ptr<IntegerRelation> cloneRel = rel.clone();
  cloneRel->projectOut(0, varIndex);
  cloneRel->projectOut(1, cloneRel->getNumVars() - 1);
  cloneRel->simplify();
  std::optional<int64_t> lower = cloneRel->getConstantBound64(BoundType::LB, 0);
  std::optional<int64_t> upper = cloneRel->getConstantBound64(BoundType::UB, 0);

  if (!lower.has_value() || !upper.has_value()) {
    return failure();
  }
  return std::make_pair(lower.value(), upper.value());
}

// A class to manage a multi-step analysis of an integer relation's tableau.
class TableauAnalysis {
 public:
  TableauAnalysis(const IntegerRelation& rel, MLIRContext* context)
      : rel(rel), context(context){};

  LogicalResult runAnalysis() {
    // The order of these steps matters. First we eliminate local variables,
    // then nonlocal vars using the expressions for eliminated local variables.
    // The remaining non-eliminated variables become induction variables.
    tryEliminateLocalVars();
    tryEliminateNonlocalVars();
    identifyInductionVars();
    if (failed(estimateProjectedBounds())) {
      return failure();
    }
    return success();
  }

  void tryEliminateLocalVars() {
    if (rel.getNumLocalVars() == 0) {
      return;
    }
    eliminatedVariables.reserve(rel.getNumLocalVars());

    // index i corresponds to the i-th variable in the relation.
    SmallVector<AffineExpr, 4> dims;
    for (unsigned i = 0; i < rel.getNumVars(); ++i) {
      dims.push_back(getAffineDimExpr(i, context));
    }

    // For each local variable, we find an equality that uses the variable and
    // isolate it.
    // FIXME: also support replacing local variables in terms of previously
    // found replacements.
    for (unsigned j = rel.getVarKindOffset(VarKind::Local);
         j < rel.getVarKindEnd(VarKind::Local); ++j) {
      // This always finds the first equality with a non-zero coefficient,
      // though perhaps we may need to find a different one, if the local
      // variable we're eliminating has a simpler equality that shows up later,
      // e.g., maybe the first equality found has another local variable in it,
      // but a later one has only domain/range vars.
      //
      // FIXME: replace this with a search for the first equality that has only
      // domain/range variables in it, or local variables that have already been
      // eliminated.
      std::optional<unsigned> nonzeroConstraintIndex =
          rel.findConstraintWithNonZeroAt(j, /*isEq=*/true);
      if (!nonzeroConstraintIndex.has_value()) {
        eliminatedVariables[j] =
            getAffineConstantExpr(0, context);  // local var == zero
      }
      SmallVector<int64_t> constraint =
          rel.getEquality64(nonzeroConstraintIndex.value());

      // The coefficient of the local variable is divided through the rest of
      // the expr. We negate the coefficient of the local variable because we
      // move it to the other side of the equality before dividing. E.g.,
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
        if (i == j) continue;  // Don't include the local variable.
        if (constraint[i] == 0) continue;
        localExpr = localExpr + dims[i] * constraint[i];
      }
      localExpr = localExpr.floorDiv(divisor);

      LLVM_DEBUG(llvm::errs() << "Identified local expr for variable " << j
                              << ": " << localExpr << "\n");

      eliminatedVariables[j] = localExpr;
    }
  }

  void tryEliminateNonlocalVars() {}

  void identifyInductionVars() {
    // The induction variables are those that are not eliminated, i.e., they
    // are not in the eliminatedVariables map.
    for (unsigned i = 0; i < rel.getNumVars(); ++i) {
      if (eliminatedVariables.count(i) == 0) {
        inductionVars.push_back(i);
      }
    }
  }

  LogicalResult estimateProjectedBounds() {
    for (unsigned i : inductionVars) {
      auto res = getProjectedBounds(rel, i);
      if (failed(res)) {
        return failure();
      }
      auto [lower, upper] = res.value();
      inductionVarBounds[i] = std::make_pair(lower, upper);
    }
    return success();
  }

  const DenseMap<unsigned, AffineExpr>& getEliminatedVariables() const {
    return eliminatedVariables;
  }

  const std::pair<int64_t, int64_t>& getInductionVarBounds(unsigned i) const {
    return inductionVarBounds.at(i);
  }

  const SmallVector<unsigned>& getInductionVars() const {
    return inductionVars;
  }

  AffineExpr convertConstraintToAffineExpr(unsigned constraintIndex,
                                           bool isEq) const {
    // Not using getAffineExprFromFlatForm because we want to allow some local
    // vars to be unresolved.
    SmallVector<int64_t> constraint =
        isEq ? rel.getEquality64(constraintIndex)
             : rel.getInequality64(constraintIndex);
    auto expr = getAffineConstantExpr(0, context);
    unsigned symbolOffset = rel.getVarKindOffset(VarKind::Symbol);

    // Variable terms
    for (unsigned j = 0; j < rel.getNumVars(); ++j) {
      if (constraint[j] == 0) continue;
      auto id = rel.getVarKindAt(j) == VarKind::Symbol
                    ? getAffineSymbolExpr(j - symbolOffset, context)
                    : getAffineDimExpr(j, context);
      expr = expr + id * constraint[j];
    }

    // Constant term.
    int64_t constTerm = constraint[constraint.size() - 1];
    if (constTerm != 0) expr = expr + constTerm;

    // For each variable, if it's been replaced, substitute it in the expr.
    SmallVector<AffineExpr> dimReplacements;
    for (const auto& [varIndex, localExpr] : eliminatedVariables) {
      if (expr.isFunctionOfDim(varIndex)) {
        LLVM_DEBUG(llvm::errs() << "Replacing variable " << varIndex
                                << " with local expr: " << localExpr << "\n");
        dimReplacements.push_back(localExpr);
      } else {
        // identity mapping
        dimReplacements.push_back(getAffineDimExpr(varIndex, context));
      }
    }

    return expr.replaceDims(dimReplacements);
  }

 private:
  DenseMap<unsigned, AffineExpr> eliminatedVariables;
  SmallVector<unsigned> inductionVars;
  DenseMap<unsigned, std::pair<int64_t, int64_t>> inductionVarBounds;
  const IntegerRelation& rel;
  MLIRContext* context;
};

FailureOr<LoopNest> generateLoopNest(const IntegerRelation& rel,
                                     MLIRContext* context) {
  std::unique_ptr<IntegerRelation> cloneRel = rel.clone();
  cloneRel->simplify();
  cloneRel->removeTrivialRedundancy();

  TableauAnalysis analysis(rel, context);
  if (failed(analysis.runAnalysis())) {
    return failure();
  }

  LoopNest nest;
  SmallVector<AffineDimExpr, 4> inductionVars;

  nest.numInductionVars = analysis.getInductionVars().size();
  for (auto i : analysis.getInductionVars()) {
    auto [lower, upper] = analysis.getInductionVarBounds(i);
    nest.lowerBounds.push_back(lower);
    nest.upperBounds.push_back(upper);
  }

  for (int i = 0; i < cloneRel->getNumEqualities(); ++i) {
    SmallVector<int64_t> constraint = cloneRel->getEquality64(i);
    AffineExpr expr = analysis.convertConstraintToAffineExpr(i, true);
    nest.constraints.push_back(expr);
    nest.eq.push_back(true);
  }

  for (int i = 0; i < cloneRel->getNumInequalities(); ++i) {
    std::cerr << "step" << i << "\n";
    SmallVector<int64_t> constraint = cloneRel->getInequality64(i);
    AffineExpr expr = analysis.convertConstraintToAffineExpr(i, false);
    nest.constraints.push_back(expr);
    nest.eq.push_back(false);
  }
  std::cerr << "step3\n";
  return nest;
}

}  // namespace heir
}  // namespace mlir
