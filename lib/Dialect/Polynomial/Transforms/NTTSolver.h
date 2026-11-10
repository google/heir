#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_

#include <cstdint>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "ortools/sat/cp_model.h"               // from @com_google_ortools

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
  // v's conversion cost in the objective is scaled by the number of times
  // v's conversion site executes (i.e., the product of the trip counts of all
  // loops enclosing it).
  RepVars& getOrCreateVars(const Value& v);
  operations_research::sat::CpModelBuilder model;
  llvm::DenseMap<Value, RepVars> vars;
  // Every mode variable created by addFloatingMode. solve() uses these to
  // break ties between equal-cost solutions in favor of COEFF mode.
  llvm::SmallVector<operations_research::sat::BoolVar> modeVars;
  // Total conversion cost. solve() minimizes this first, then the tie-break.
  operations_research::sat::LinearExpr objective;

 public:
  void forceDemandEitherForm(const Value& v);
  void forceDemandFixedForm(const Value& v, Form form);
  void implyForm(const Value& v, Form a, Form b);
  void implyUse(const Value& out, const Value& in, Form form);
  // `out`'s mode (0 = COEFF, 1 = EVAL) determines which form `in` must supply,
  // gated on `out` being used in some form at all (if `out` is dead, nothing
  // is forced on `in`). Also used to tie a region-branch successor input's
  // shared mode (see addFloatingMode) to the operand forwarding into it.
  void implyMode(const Value& out, const Value& in);
  void prohibitBothForms(const Value& v);
  void addConversionCostForForm(const Value& v, Form form);
  // Allocates one fresh mode variable (0 = COEFF, 1 = EVAL) shared by every
  // value in `members`, and returns it. The mode is left fully free -- driven
  // only by the objective via addModeConversionCost -- since any value in
  // `members` may have its own, independently-forced demand that the shared
  // mode has to reconcile rather than dictate. For a single-output "either
  // form" op, call this with a one-element list; for a region-branch
  // successor-input group (e.g. a loop's iter_arg and its result, which MLIR
  // requires to share one materialized type), call it with every member of
  // that group.
  operations_research::sat::BoolVar addFloatingMode(
      llvm::ArrayRef<Value> members);
  // Charges `member`'s conversion variable iff `member`'s own demand
  // disagrees with the shared `mode`: conv == (mode ∧ c) ∨ (¬mode ∧ e). This
  // generalizes "conversion needed iff both forms independently demanded" to
  // the case where `member` shares its native form with other values via
  // addFloatingMode and may need a conversion even when its own demand is
  // one-sided, because a different member of the group pulled the shared mode
  // the other way.
  void addModeConversionCost(const Value& member,
                             const operations_research::sat::BoolVar& mode);
  void setZeroConversionCost(const Value& v);
  // Returns a minimum-conversion-cost solution. Solutions are not guaranteed
  // to be unique or repeatable.
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
  // True if CP-SAT proved the solution optimal or feasible; anything else is an
  // error.
  bool isValid() const;
  // True iff CP-SAT proved the solution optimal.
  bool isOptimal() const;

 private:
  const NTTSolver& solver;
  const operations_research::sat::CpSolverResponse soln;
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_
