#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project
#include "ortools/sat/cp_model.h"            // from @com_google_ortools
#include "ortools/sat/cp_model_solver.h"     // from @com_google_ortools

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
  RepVars& getOrCreateVars(Value& v);
  const RepVars& getVars(Value& v) const;
  operations_research::sat::CpModelBuilder model;
  llvm::DenseMap<Value, RepVars> vars;
  operations_research::sat::LinearExpr objective;

 public:
  void allowEitherForm(Value& v);
  void fixForm(Value& v, Form form);
  void implyForm(Value& v, Form a, Form b);
  void implyUse(Value& out, Value& in, Form form);
  void implyMode(Value& out, Value& in);
  void prohibitBothForms(Value& v);
  void addConversionCostForForm(Value& v, Form form);
  void addConversionCostIfBothForms(Value& v);
  void setZeroConversionCost(Value& v);
  void addOpMode(Value& v);
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

  bool needsForm(Value& v, Form form) const;
  bool needsConversion(Value& v) const;
  Form getMode(Value& v) const;
  bool isValid() const;

 private:
  const NTTSolver& solver;
  const operations_research::sat::CpSolverResponse soln;
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTT_SOLVER_H_
