#include "lib/Utils/Layout/Codegen.h"

#include <iostream>

#include "llvm/include/llvm/ADT/DenseMap.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"    // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExprVisitor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project

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
      LLVM_DEBUG(llvm::dbgs() << "Converted equality " << i
                              << " to affine expr: " << expr << " == 0\n");
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
      LLVM_DEBUG(llvm::dbgs() << "Converted inequality " << i
                              << " to affine expr: " << expr << " >= 0\n");
      resultNest->loops.back().constraints.push_back(expr);
      resultNest->loops.back().eq.push_back(false);
    }

    // Step 5: AffineExpr recognizes modulos from divs, and lifts them
    // appropriately. So now we can identify any constraints in the innermost
    // loop that are of the form (c * v0 + foo) % n == 0, where foo does not
    // depend on v0, and eliminate the loop iterating over v0 while adding a
    // new eliminated variable to the prior loop.
    do {
      llvm::errs() << "Finding plain modulo equality to eliminate\n";
      auto res = findPlainModuloEquality();
      if (failed(res)) {
        llvm::errs() << "Failed to find a plain modulo equality.\n";
        break;  // No more eliminatable constraints.
      }
      auto [varIndex, expr] = res.value();
      llvm::errs() << "Found plain modulo equality for var " << varIndex << ": "
                   << expr << "\n";

      // Find the index of the loop in resultNests->loops that has this var
      // as its induction variable.
      unsigned loopIndex = -1;
      for (unsigned i = 0; i < resultNest->loops.size(); ++i) {
        if (resultNest->loops[i].inductionVar == varIndex) {
          loopIndex = i;
          break;
        }
      }
      assert(loopIndex > 1 &&
             "The found variable must be an induction variable in some "
             "loop after the first");

      eliminatedVariables[varIndex] = expr;
      resultNest->loops[loopIndex - 1].eliminatedVariables[varIndex] = expr;
      resultNest->loops.erase(resultNest->loops.begin() + loopIndex);
      inductionVars.erase(
          std::remove(inductionVars.begin(), inductionVars.end(), varIndex),
          inductionVars.end());
      llvm::errs() << "Loop nest is now:\n";
      for (const auto& loop : resultNest->loops) {
        std::cerr << "  " << loop << "\n";
      }
    } while (succeeded(findPlainModuloEquality()));

    return success();
  }

  // Finds a variable that can be eliminated as a plain modulo equality, and
  // returns a pair of the variable's index in `rel` and the isolated
  // expression, or a failure if no such variable can be found/eliminated.
  FailureOr<std::pair<unsigned, AffineExpr>> findPlainModuloEquality() const {
    auto constraints = resultNest->loops.back().constraints;
    auto eq = resultNest->loops.back().eq;
    for (unsigned i = 0; i < constraints.size(); ++i) {
      llvm::errs() << "Checking constraint " << i << ": " << constraints[i]
                   << "\n";
      if (!eq[i]) {
        continue;
      }

      // Check if the constraint is of the form (c * v0 + foo) % n == 0,
      // where foo does not depend on v0.
      AffineExpr expr = constraints[i];
      unsigned numDims = rel.getNumVars();
      SimpleAffineExprFlattener flattener(numDims, /*numSymbols=*/0);
      if (failed(flattener.walkPostOrder(expr))) {
        llvm::errs() << "Failed to flatten\n.";
        return failure();
      }

      SmallVector<int64_t, 8> flattenedExpr(flattener.operandExprStack.back());
      llvm::errs() << "Checking constraint: " << expr << " flattened to: [";
      for (unsigned j = 0; j < flattenedExpr.size(); ++j) {
        llvm::errs() << flattenedExpr[j];
        if (j < flattenedExpr.size() - 1) {
          llvm::errs() << ", ";
        }
      }
      llvm::errs() << "]\n";

      // Check if the constraint has an eliminatable domain variable
      std::optional<unsigned> chosenVar = std::nullopt;
      for (unsigned varIndex = rel.getVarKindOffset(VarKind::Domain);
           varIndex < rel.getVarKindEnd(VarKind::Domain); ++varIndex) {
        llvm::errs() << "Checking flattened constraint for domain var: "
                     << varIndex << " with coeff: " << flattenedExpr[varIndex]
                     << "\n";
        if (flattenedExpr[varIndex] != 0 &&
            !eliminatedVariables.count(varIndex)) {
          // // Lastly, we need to ensure the domain variable in question
          // // is not in any of the localExprs from the flattening pass.
          // // If it is, then we cannot trivially isolate it.
          // FIXME: what to do here???
          // llvm::all_of(flattener.localExprs,
          //              [varIndex](AffineExpr localExpr) {
          //                if (localExpr.isFunctionOfDim(varIndex)) {
          //                  llvm::errs() << "  Local expr " << localExpr
          //                               << " is a function of var "
          //                               << varIndex << ", skipping\n";
          //                  return false;  // Cannot isolate this var.
          //                }
          //                return !localExpr.isFunctionOfDim(varIndex);
          //              })) {
          chosenVar = varIndex;
          break;
        }
      }

      if (!chosenVar.has_value()) {
        continue;  // No eliminatable domain variable found.
      }

      // To isolate the variable, we are converting an expression like
      //
      //   (3a + b) % 5 == 0  <--> 3a + b - 5q == 0
      //
      // to
      //
      //   3a == -b + 5q
      //
      // We can isolate without dividing in the flattened form, then
      // recosntruct the AffineExpr before doing a floorDiv.
      unsigned varIndex = chosenVar.value();
      int64_t coeff = flattenedExpr[varIndex];
      for (long& coeffIndex : flattenedExpr) {
        coeffIndex = -coeffIndex;
      }
      flattenedExpr[varIndex] = 0;

      AffineExpr reconstructed = getAffineExprFromFlatForm(
          flattenedExpr, rel.getNumVars(), 0, flattener.localExprs, context);
      AffineExpr isolated = reconstructed.floorDiv(coeff);
      return std::make_pair(varIndex, isolated);
    }

    return failure();
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
          LLVM_DEBUG(llvm::dbgs() << "Identified local expr for variable "
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

    // Variable terms
    for (unsigned j = 0; j < rel.getNumVars(); ++j) {
      if (constraint[j] == 0) continue;
      assert(rel.getVarKindAt(j) != VarKind::Symbol &&
             "Symbols are not supported in layout codegen");
      auto id = getAffineDimExpr(j, context);
      expr = expr + id * constraint[j];
    }

    // Constant term.
    int64_t constTerm = constraint[constraint.size() - 1];
    if (constTerm != 0) expr = expr + constTerm;

    // For each variable, if it's been replaced, substitute it in the expr.
    SmallVector<AffineExpr> dimReplacements;

    for (unsigned varIndex = 0; varIndex < rel.getNumVars(); ++varIndex) {
      if (eliminatedVariables.count(varIndex) &&
          expr.isFunctionOfDim(varIndex)) {
        AffineExpr localExpr = eliminatedVariables.at(varIndex);
        dimReplacements.push_back(localExpr);
      } else {
        // identity mapping
        dimReplacements.push_back(getAffineDimExpr(varIndex, context));
      }
    }

    AffineExpr result = expr.replaceDims(dimReplacements);
    return simplifyAffineExpr(result, rel.getNumVars(), 0);
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

  // Get the IntMatrix underneath the relation:
  presburger::IntMatrix matrix = cloneRel->getEqualities();
  matrix.dump();
  auto [H, U] = matrix.computeHermiteNormalForm();
  H.dump();
  U.dump();

  LoopNest resultNest;
  LoopNestGenerator gen(&resultNest, *cloneRel, context);
  if (failed(gen.run())) return failure();
  return resultNest;
}

}  // namespace heir
}  // namespace mlir
