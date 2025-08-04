#include "lib/Utils/Layout/Codegen.h"

#include "llvm/include/llvm/ADT/DenseMap.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"    // from @llvm-project
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

bool isConstantBound(ArrayRef<int64_t> constraint) {
  // The last index is the constant term, so check that only one other value
  // is nonzero.
  return llvm::count_if(llvm::drop_end(constraint),
                        [](int64_t v) { return v != 0; }) == 1;
}

// Generate a perfect loop nest from an integer relation. Makes assumptions
// about the structure of the relation vis-a-vis HEIR's ciphertext packing.
class LoopNestGenerator {
 public:
  LoopNestGenerator(LoopNest* resultNest, const IntegerRelation& rel,
                    MLIRContext* context)
      : resultNest(resultNest), rel(rel), context(context){};

  LogicalResult run() {
    // We are free to pick the order of the induction variables in the loop,
    // and we happen to know that layouts defined by valid ciphertext packings
    // are partial functions from the range to the domain. So in the best case,
    // we can iterate over the range varianges (which we generally will have to
    // do), and then eliminate all other variables, leaving us with the optimal
    // loop nest.
    //
    // Step 1: turn each range variable into an induction variable.
    for (unsigned i = rel.getVarKindOffset(VarKind::Range);
         i < rel.getVarKindEnd(VarKind::Range); ++i) {
      inductionVars.push_back(i);
      auto res = getProjectedBounds(rel, i);
      if (failed(res)) {
        return failure();
      }
      auto bounds = res.value();
      resultNest->addLoop(i, bounds.first, bounds.second);
    }

    while (moreVariablesToProcess()) {
      // Step 2: determine if any variables can now be written purely in terms
      // of the induction variables so far. If one can add an expression for
      // the eliminated variable (which will be materialized as an arithmetic
      // statement in the loop) and then keep doing this until no more
      // variables remain or none can be eliminated.
      bool changed = true;
      while (changed && moreVariablesToProcess()) {
        changed = tryEliminateVars();
      }

      if (!moreVariablesToProcess()) {
        // No more variables to process, we are done.
        break;
      }

      // Step 3: Pick the next variable to be an induction variable and repeat.
      unsigned nextVar = findFallbackInductionVar();
      inductionVars.push_back(nextVar);
      auto res = getProjectedBounds(rel, nextVar);
      if (failed(res)) {
        return failure();
      }
      auto bounds = res.value();
      resultNest->addLoop(nextVar, bounds.first, bounds.second);
    }

    // Step 4: Check the constraints for the inner-most loop body.
    // Skip trivial constraints that are implied by bounds checking.
    for (unsigned i = 0; i < rel.getNumEqualities(); ++i) {
      SmallVector<int64_t> equality = rel.getEquality64(i);
      if (isConstantBound(equality)) {
        continue;
      }
      AffineExpr expr = convertConstraintToAffineExpr(i, true);
      if (expr.isSymbolicOrConstant()) {
        continue;
      }
      resultNest->loops.back().constraints.push_back(expr);
      resultNest->loops.back().eq.push_back(true);
    }
    for (unsigned i = 0; i < rel.getNumInequalities(); ++i) {
      SmallVector<int64_t> inequality = rel.getInequality64(i);
      if (isConstantBound(inequality)) {
        continue;
      }
      AffineExpr expr = convertConstraintToAffineExpr(i, false);
      if (expr.isSymbolicOrConstant()) {
        continue;
      }
      resultNest->loops.back().constraints.push_back(expr);
      resultNest->loops.back().eq.push_back(false);
    }

    return success();
  }

  inline bool moreVariablesToProcess() const {
    return inductionVars.size() + eliminatedVariables.size() < rel.getNumVars();
  }

  // Find the first variable that is not an induction variable or eliminated.
  inline unsigned findFallbackInductionVar() const {
    for (unsigned i = 0; i < rel.getNumVars(); ++i) {
      if (eliminatedVariables.count(i) == 0 &&
          !llvm::is_contained(inductionVars, i)) {
        return i;
      }
    }
    llvm_unreachable("No more variables to process");
    return -1;
  }

  FailureOr<AffineExpr> tryIsolateVar(unsigned varIndex,
                                      const SmallVector<int64_t>& equality) {
    // Try to isolate the variable at `varIndex` in the given equality.
    // If it is not possible, return failure.
    //
    // The coefficient of the variable is divided through the rest of the expr.
    // We negate the coefficient of the variable because we move it to the
    // other side of the equality before dividing. E.g.,
    //
    // Tableau row 1  -1  -2  0  == 0 corresponds to d0 - d1 - 2q + 0 == 0
    //
    // And to isolate q we convert it to
    //
    //   d0 - d1 == 2q
    //
    // and divide through by 2.
    int64_t divisor = -equality[varIndex];
    if (divisor == 0) {
      return failure();
    }

    AffineExpr expr = getAffineConstantExpr(0, context);
    for (unsigned i = 0; i < equality.size(); ++i) {
      if (i == varIndex) continue;     // Skip the variable we are isolating.
      if (equality[i] == 0) continue;  // Skip zero coefficients.
      expr = expr + getAffineDimExpr(i, context) * equality[i];
    }
    expr = expr.floorDiv(divisor);

    // If the expr can be written purely in terms of eliminated variables and
    // induction variables, then we can isolate it. Test this by replacing all
    // eliminated dims with zero, and checking if the affine expr is constant.
    SmallVector<AffineExpr> dimReplacements;
    for (unsigned i = 0; i < rel.getNumVars(); ++i) {
      if (eliminatedVariables.count(i) > 0 ||
          llvm::is_contained(inductionVars, i)) {
        dimReplacements.push_back(getAffineConstantExpr(0, context));
      } else {
        // Do not replace anything else
        dimReplacements.push_back(getAffineDimExpr(i, context));
      }
    }
    AffineExpr testExpr = expr.replaceDims(dimReplacements);
    if (testExpr.isSymbolicOrConstant()) {
      return expr;
    }

    return failure();
  }

  bool tryEliminateVars() {
    // index i corresponds to the i-th variable in the relation.
    SmallVector<AffineExpr, 4> dims;
    for (unsigned i = 0; i < rel.getNumVars(); ++i) {
      dims.push_back(getAffineDimExpr(i, context));
    }

    // For each variable, try to find an equality that uses the variable and
    // isolate it.
    for (unsigned varIndex = 0; varIndex < rel.getNumVars(); ++varIndex) {
      if (eliminatedVariables.count(varIndex) > 0 ||
          llvm::is_contained(inductionVars, varIndex)) {
        // Already eliminated or is an induction variable.
        continue;
      }

      for (unsigned constraintIndex = 0;
           constraintIndex < rel.getNumEqualities(); ++constraintIndex) {
        SmallVector<int64_t> equality = rel.getEquality64(constraintIndex);
        if (equality[varIndex] == 0) {
          // This equality does not use the variable.
          continue;
        }
        FailureOr<AffineExpr> res = tryIsolateVar(varIndex, equality);
        if (succeeded(res)) {
          AffineExpr localExpr = res.value();
          LLVM_DEBUG(llvm::errs() << "Identified local expr for variable "
                                  << varIndex << ": " << localExpr << "\n");
          eliminatedVariables[varIndex] = localExpr;
          resultNest->loops.back().eliminatedVariables[varIndex] = localExpr;
          return true;
        }
      }
    }

    return false;
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
    for (unsigned varIndex = 0; varIndex < rel.getNumVars(); ++varIndex) {
      if (eliminatedVariables.count(varIndex)) {
        if (expr.isFunctionOfDim(varIndex)) {
          AffineExpr localExpr = eliminatedVariables.at(varIndex);
          dimReplacements.push_back(localExpr);
        }
      } else {
        // identity mapping
        dimReplacements.push_back(getAffineDimExpr(varIndex, context));
      }
    }

    return expr.replaceDims(dimReplacements);
  }

 private:
  // Pointer to the output loop nest, provided by the caller.
  LoopNest* resultNest;

  DenseMap<unsigned, AffineExpr> eliminatedVariables;
  SmallVector<unsigned> inductionVars;
  const IntegerRelation& rel;
  MLIRContext* context;
};

FailureOr<LoopNest> generateLoopNest(const IntegerRelation& rel,
                                     MLIRContext* context) {
  std::unique_ptr<IntegerRelation> cloneRel = rel.clone();
  // This does a basic simplification. In the future, we may want to do a more
  // sophisticated simplification, such as reducing the contsraint matrix to
  // Hermite Normal Form, which would allow for the elimination of more
  // variables. MLIR has IntMatrix::computeHermiteNormalForm, but I (j2kun@)
  // haven't tried it yet, and it's not clear to me if it will properly
  // preserve Identifiers attached to the variables, which we need to recover
  // SSA values from the reduced matrix.
  cloneRel->simplify();
  cloneRel->removeTrivialRedundancy();
  cloneRel->dump();

  LoopNest resultNest;
  LoopNestGenerator gen(&resultNest, *cloneRel, context);
  if (failed(gen.run())) return failure();
  return resultNest;
}

}  // namespace heir
}  // namespace mlir
