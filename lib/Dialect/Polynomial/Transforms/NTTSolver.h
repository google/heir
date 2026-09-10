#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_

#include <cstdint>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project
#include "ortools/sat/cp_model.h"            // from @com_google_ortools

namespace mlir {
namespace heir {
namespace polynomial {

class CPSATSolution;

// This class translates domain logic (i.e., a Polynomial AST) into CP-SAT logic
class NTTSolver {
 private:
  struct RepVars {
    // 1 if v is needed in coefficient form, 0 otherwise
    operations_research::sat::BoolVar c;
    // 1 if v is needed in evaluation form, 0 otherwise
    operations_research::sat::BoolVar e;
    // 1 if a conversion is needed for this value, 0 otherwise
    operations_research::sat::BoolVar conv;
    // for nodes that work in either form, this variable is 0 for
    // coeff mode and 1 for eval mode
    operations_research::sat::BoolVar mode;

    const operations_research::sat::BoolVar& getVarForm(Form form) const;
  };
  RepVars& getOrCreateVars(const Value& v);
  operations_research::sat::CpModelBuilder model;
  llvm::DenseMap<Value, RepVars> vars;
  llvm::DenseMap<Value, int64_t> conversionCostMultipliers;
  operations_research::sat::LinearExpr objective;

 public:
  // Scales v's conversion cost in the objective by `multiplier` (e.g. the
  // number of times a loop containing v's conversion site will execute).
  // Must be called before any other solver method touches v.
  void setConversionCostMultiplier(const Value& v, int64_t multiplier);
  void forceDemandEitherForm(const Value& v);
  void forceDemandFixedForm(const Value& v, Form form);
  void implyForm(const Value& v, Form a, Form b);
  void implyUse(const Value& out, const Value& in, Form form);
  // Requires `source` to supply whichever form `target` is materialized in
  // natively -- COEFF if target's coeff-demand bit is set, EVAL otherwise
  // (mirroring the "needsForm(COEFF) ? COEFF : EVAL" tie-break PolyMulToNTT.cpp
  // uses when actually materializing a value). This is weaker than requiring
  // `source` to supply *every* form `target` needs: if target additionally
  // needs the other form too, that is satisfied by a separate, locally
  // materialized conversion at target's own definition site, which does not
  // require `source` to supply it. Forwarding edges into a region-branch
  // successor input (e.g. a loop's entry operand into its iter_arg) use this
  // instead of two implyUse calls to avoid over-constraining `source` into
  // needing a form nothing actually consumes.
  void requireSourceMatchesNativeForm(const Value& target, const Value& source);
  void implyMode(const Value& out, const Value& in);
  void prohibitBothForms(const Value& v);
  // Forces two values to resolve to the same *native* (materialized-in-the-IR)
  // form. This is for region-branch successor inputs (e.g. a loop iter_arg)
  // that share a single physical operand: MLIR's RegionBranchOpInterface can
  // forward one operand to several successor inputs at once (e.g. scf.for's
  // scf.yield operand doubles as both the next iteration's iter_arg and the
  // loop's own result), so those targets have no independent operand slot to
  // diverge on and must end up with the same materialized type.
  //
  // This only needs to tie the coeff-demand bit, not the full demand pattern:
  // the native form of any value in this pass is chosen as
  // "needsForm(COEFF) ? COEFF : EVAL" (see PolyMulToNTT.cpp), a function of
  // the coeff-demand bit alone. Tying just that bit is therefore sufficient to
  // guarantee the two values resolve to the same materialized type, while
  // leaving each value free to independently need (or not need) the other
  // form as a separate, locally materialized conversion -- e.g. a loop
  // iter_arg used in eval form only inside the loop body shouldn't force the
  // loop's result to also be materialized in eval form if nothing outside the
  // loop needs it.
  void equateNativeForm(const Value& a, const Value& b);
  void addConversionCostForForm(const Value& v, Form form);
  void addConversionCostIfBothForms(const Value& v);
  void setZeroConversionCost(const Value& v);
  void addOpMode(const Value& v);
  CPSATSolution solve();
  friend class CPSATSolution;
};

// Similar to NTTSolver, this class translates from the CP-SAT solution
// to problem-domain APIs
class CPSATSolution {
 public:
  explicit CPSATSolution(
      const NTTSolver& solver,
      const operations_research::sat::CpSolverResponse& soln);

  bool needsForm(const Value& v, Form form) const;
  bool needsConversion(const Value& v) const;
  Form getMode(const Value& v) const;
  bool isValid() const;

 private:
  const NTTSolver& solver;
  const operations_research::sat::CpSolverResponse soln;
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_
