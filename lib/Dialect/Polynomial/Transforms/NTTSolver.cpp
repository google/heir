#include "lib/Dialect/Polynomial/Transforms/NTTSolver.h"

#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

using namespace operations_research;
using namespace operations_research::sat;

CPSATSolution::CPSATSolution(const NTTSolver& solver,
                             const CpSolverResponse& soln)
    : solver(solver), soln(soln) {}

bool CPSATSolution::needsForm(Value& v, Form form) const {
  const BoolVar& bv = solver.getVars(v).getVarForm(form);
  return SolutionBooleanValue(soln, bv);
}

bool CPSATSolution::needsConversion(Value& v) const {
  const NTTSolver::RepVars& vs = solver.getVars(v);
  return SolutionBooleanValue(soln, vs.conv);
}

Form CPSATSolution::getMode(Value& v) const {
  const NTTSolver::RepVars& vs = solver.getVars(v);
  return SolutionBooleanValue(soln, vs.mode) ? Form::EVAL : Form::COEFF;
}

bool CPSATSolution::isValid() const {
  return soln.status() == CpSolverStatus::OPTIMAL ||
         soln.status() == CpSolverStatus::FEASIBLE;
}

int getConversionCost(Value& v) {
  Type t = v.getType();
  if (auto p = dyn_cast<PolynomialType>(t)) {
    return 1;
  }
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    return rt.getNumElements();
  }
  return 0;
}

const NTTSolver::RepVars& NTTSolver::getVars(Value& v) const {
  auto it = vars.find(v);
  if (it != vars.end()) {
    return it->second;
  } else {
    llvm_unreachable("Var does not exist");
  }
}

NTTSolver::RepVars& NTTSolver::getOrCreateVars(Value& v) {
  auto it = vars.find(v);
  if (it != vars.end()) {
    return it->second;
  }
  int convCost = getConversionCost(v);
  RepVars repVars{/*c=*/model.NewBoolVar(),
                  /*e=*/model.NewBoolVar(),
                  /*conv=*/model.NewBoolVar(),
                  /*mode=*/BoolVar()};

  objective += repVars.conv;
  // add a new conversion variable equal to the "representative" conversion cost
  // and force equality between them. We never need references to these other
  // variables though; we just use repVars.conv as their proxy.
  for (int i = 1; i < convCost; i++) {
    BoolVar b = model.NewBoolVar();
    model.AddEquality(repVars.conv, b);
    objective += b;
  }
  vars[v] = repVars;
  return vars[v];
}

const BoolVar& NTTSolver::RepVars::getVarForm(Form form) const {
  return form == Form::COEFF ? c : e;
}

void NTTSolver::allowEitherForm(Value& v) {
  RepVars& vs = getOrCreateVars(v);
  model.AddBoolOr({vs.c, vs.e});
}

void NTTSolver::fixForm(Value& v, Form form) {
  RepVars& vs = getOrCreateVars(v);
  model.AddEquality(vs.getVarForm(form), 1);
}

void NTTSolver::implyForm(Value& v, Form a, Form b) {
  RepVars& vs = getOrCreateVars(v);
  model.AddImplication(vs.getVarForm(a), vs.getVarForm(b));
}

void NTTSolver::prohibitBothForms(Value& v) {
  RepVars& vs = getOrCreateVars(v);
  model.AddBoolOr(
      {vs.getVarForm(Form::COEFF).Not(), vs.getVarForm(Form::EVAL).Not()});
}

void NTTSolver::implyUse(Value& out, Value& in, Form form) {
  RepVars& outs = getOrCreateVars(out);
  RepVars& ins = getOrCreateVars(in);
  model.AddImplication(outs.getVarForm(form), ins.getVarForm(form));
}

void NTTSolver::implyMode(Value& out, Value& in) {
  RepVars& outs = getOrCreateVars(out);
  RepVars& ins = getOrCreateVars(in);
  // if mode = 0 and output in any form is needed, force inputs to coeff
  // ¬v_mode ^ v_c => u_c
  model.AddBoolOr({outs.mode, outs.c.Not(), ins.c});
  // ¬v_mode ^ v_e => u_c
  model.AddBoolOr({outs.mode, outs.e.Not(), ins.c});

  // if mode = 1 and output in any form is needed, force inputs to eval
  // v_mode ^ v_c => u_c
  model.AddBoolOr({outs.mode.Not(), outs.c.Not(), ins.e});
  // v_mode ^ v_e => u_c
  model.AddBoolOr({outs.mode.Not(), outs.e.Not(), ins.e});
}

void NTTSolver::addConversionCostForForm(Value& v, Form form) {
  RepVars& vs = getOrCreateVars(v);
  model.AddEquality(vs.conv, form == Form::COEFF ? vs.c : vs.e);
}

void NTTSolver::addConversionCostIfBothForms(Value& v) {
  RepVars& vs = getOrCreateVars(v);
  // vs.conv <=> vs.c and vs.e (otherwise we just run the op in the mode
  // required)

  // The first two give vs.conv => vs.c and vs.conv => vs.e, hence vs.conv =>
  // vs.c ^ vs.e
  model.AddImplication(vs.conv, vs.c);
  model.AddImplication(vs.conv, vs.e);
  // The last is vs.c ^ vs.e => vs.conv
  model.AddBoolOr({vs.c.Not(), vs.e.Not(), vs.conv});
}

void NTTSolver::setZeroConversionCost(Value& v) {
  RepVars& vs = getOrCreateVars(v);
  model.AddEquality(vs.conv, 0);
}

void NTTSolver::addOpMode(Value& v) {
  RepVars& vs = getOrCreateVars(v);
  vs.mode = model.NewBoolVar();
  // 1 = E-mode, 0 = C-mode
  // If v_e is needed but not v_c, set v_mode = 1
  // v_e ^ ¬v_c => v_mode
  model.AddBoolOr({vs.e.Not(), vs.c, vs.mode});
  // If v_c is needed but not v_e, set v_mode = 0
  // v_c ^ ¬v_e => v_mode
  model.AddBoolOr({vs.c.Not(), vs.e, vs.mode.Not()});
}

CPSATSolution NTTSolver::solve() {
  model.Minimize(objective);
  const CpSolverResponse resp = Solve(model.Build());
  return CPSATSolution(*this, resp);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
