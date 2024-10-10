#include "lib/Analysis/OptimizeRelinearizationAnalysis/OptimizeRelinearizationAnalysis.h"

#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "ortools/linear_solver/linear_solver.h"  // from @com_google_ortools

// The key basis degree describes the highest degree in the key basis, i.e., a
// key basis of (1, s, s^2) has degree 2, while (1, s, s^2, s^3) has degree 3.
// This implementation makes some simplifying assumptions: that the key basis
// is always of the form (1, s, s^2, ..., s^d), and that the key basis degree
// of a rotation op input/output is always 1. It is possible to implement
// multiplication for an arbitrary key basis, as well as implement rotation
// without key switching, but this would lead to key bases of the form (1, s^i)
// and the ILP would likely have to track additional variables to understand
// the full key basis. Meanwhile, most library implementations force the
// assumptions made here anyway.
//
// For now, fix the key basis degree bound to 3. Could make it a pass flag
// later, or introduce costs for high-degree operations and let it be
// unconstrained.
constexpr int MAX_KEY_BASIS_DEGREE = 3;
constexpr int INITIAL_KEY_BASIS_DEGREE = 1;
constexpr int IF_THEN_AUX = 100;

using namespace operations_research;

namespace mlir {
namespace heir {

#define DEBUG_TYPE "optimize-relinearization-analysis"

std::string nameAndLoc(Operation *op) {
  std::string varName;
  llvm::raw_string_ostream ss(varName);
  ss << op->getName() << "_" << op->getLoc();
  return ss.str();
}

bool hasCiphertextType(ValueRange range) {
  return llvm::any_of(range, [](Value value) {
    return isa<lwe::RLWECiphertextType>(value.getType());
  });
}

LogicalResult OptimizeRelinearizationAnalysis::solve() {
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  MPObjective *const objective = solver->MutableObjective();
  objective->SetMinimization();

  // Map an operation to a decision to relinearize its results.
  llvm::DenseMap<Operation *, MPVariable *> decisionVariables;
  // keyBasisArgVars maps SSA values to variables tracking the key basis degree
  // of the ciphertext at that point in the computation. If the SSA value is
  // the result of an op, this variable corresponds to the degree _after_ the
  // decision to relinearize is applied.
  llvm::DenseMap<Value, MPVariable *> keyBasisVars;
  // keyBasisResultVarsAfterRelin is the same as keyBasisArgVars, but _before_
  // the decision to relinearize is applied. We need both because the
  // post-processing of the solution requires us to remember the before-relin
  // key basis degree. We could recompute it later, but it's more general to
  // track it.
  llvm::DenseMap<Value, MPVariable *> beforeRelinVars;

  // First create a variable for each SSA value tracking the key basis degree
  // of the ciphertext at that point in the computation, as well as the decision
  // variable to track whether to insert a relinearization operation after the
  // operation.
  opToRunOn->walk([&](Operation *op) {
    std::string name = nameAndLoc(op);

    if (isa<ModuleOp>(op)) {
      return;
    }

    if (hasCiphertextType(op->getResults())) {
      auto *decisionVar = solver->MakeIntVar(0, 1, "InsertRelin_" + name);
      decisionVariables.insert(std::make_pair(op, decisionVar));
    }

    // Except for block arguments, SSA values are created as results of
    // operations. Create one keyBasisDegree variable for each op result.
    std::string varName = "Degree_" + name;
    for (Value result : op->getResults()) {
      if (!isa<lwe::RLWECiphertextType>(result.getType())) {
        continue;
      }

      auto *keyBasisVar = solver->MakeNumVar(0, MAX_KEY_BASIS_DEGREE, varName);
      keyBasisVars.insert(std::make_pair(result, keyBasisVar));

      // br means "before relin"
      std::string brVarName = varName + "_br";
      auto *brKeyBasisVar =
          solver->MakeNumVar(0, MAX_KEY_BASIS_DEGREE, brVarName);
      beforeRelinVars.insert(std::make_pair(result, brKeyBasisVar));
    }

    // Handle block arguments to the op, which are assumed to already be
    // linearized, though this could be generalized to read the degree from the
    // type.
    if (op->getNumRegions() == 0) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Handling block arguments for " << op->getName() << "\n");
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isa<lwe::RLWECiphertextType>(arg.getType())) {
            continue;
          }

          std::stringstream ss;
          ss << "Degree_ba" << arg.getArgNumber() << "_" << name;
          std::string varName = ss.str();
          auto *keyBasisVar =
              solver->MakeNumVar(0, MAX_KEY_BASIS_DEGREE, varName);
          keyBasisVars.insert(std::make_pair(arg, keyBasisVar));
        }
      }
    }
  });

  // The objective is to minimize the number of relinearization ops.
  // TODO(#1018): improve the objective function to account for differing
  // costs of operations at varying degrees.
  for (auto item : decisionVariables) {
    objective->SetCoefficient(item.second, 1);
  }

  // Constraints to initialize the key basis degree variables at the start of
  // the computation.
  for (auto &[value, var] : keyBasisVars) {
    if (llvm::isa<BlockArgument>(value)) {
      auto type = cast<lwe::RLWECiphertextType>(value.getType());
      // If the dimension is 3, the key basis is [0, 1, 2] and the degree is 2.
      int constrainedDegree = type.getRlweParams().getDimension() - 1;
      MPConstraint *const ct =
          solver->MakeRowConstraint(constrainedDegree, constrainedDegree, "");
      ct->SetCoefficient(var, 1);
    }
  }

  // For each operation, constrain its inputs to all have the same key basis
  // degree.
  std::string cstName;
  opToRunOn->walk([&](Operation *op) {
    if (op->getNumOperands() <= 1) {
      return;
    }

    std::string name = nameAndLoc(op);
    auto *anchorVar = keyBasisVars.lookup(op->getOperand(0));

    // degree(operand 0) == degree(operand i)
    for (OpOperand &opOperand : op->getOpOperands()) {
      auto *operandDegreeVar = keyBasisVars.lookup(opOperand.get());
      if (!operandDegreeVar || anchorVar == operandDegreeVar) {
        continue;
      }
      std::stringstream ss;
      ss << "ArgKeyBasisEquality_" << opOperand.getOperandNumber() << "_"
         << name;
      MPConstraint *const ct = solver->MakeRowConstraint(0, 0, ss.str());
      ct->SetCoefficient(anchorVar, -1);
      ct->SetCoefficient(operandDegreeVar, 1);
    }
  });

  // Some ops require a linear key basis. Return is a special case
  // where we require returned values from funcs to be linearized.
  opToRunOn->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation &>(*op).Case<bgv::RotateOp, func::ReturnOp>(
        [&](auto op) {
          for (Value operand : op->getOperands()) {
            auto *operandDegreeVar = keyBasisVars.lookup(operand);
            if (!operandDegreeVar) {
              // This could happen if you return a block argument without doing
              // anything to it. No variables are created, but it does not
              // necessarily need to be constrained.
              if (isa<func::ReturnOp>(op)) return;

              assert(false && "Operand not found in keyBasisVars");
            }
            cstName = "RequireLinearized_" + nameAndLoc(op);
            MPConstraint *const ct = solver->MakeRowConstraint(1, 1, cstName);
            ct->SetCoefficient(operandDegreeVar, 1);
          }
        });
  });

  // Add constraints that control the effect of relinearization insertion.
  opToRunOn->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation &>(*op)
        .Case<bgv::MulOp>([&](auto op) {
          // result_degree =
          //   (arg1_degree + arg2_degree) (1 - insert_relin_op)
          //   + 1 * insert_relin_op,
          //
          // but linearized due to the quadratic term input_key_basis_degree *
          // insert_relin_op, and an extra variable inserted to keep track of
          // the difference between the before_relin and after_relin degrees:
          //
          // before_relin = arg1_degree + arg2_degree
          // result_degree =
          //   before_relin (1 - insert_relin_op) + 1 * insert_relin_op

          auto inf = solver->infinity();
          auto lhsDegreeVar = keyBasisVars.lookup(op.getLhs());
          auto rhsDegreeVar = keyBasisVars.lookup(op.getRhs());
          auto resultBeforeRelinVar = beforeRelinVars.lookup(op.getResult());
          auto resultAfterRelinVar = keyBasisVars.lookup(op.getResult());
          auto insertRelinOpDecision = decisionVariables.lookup(op);

          std::string opName = nameAndLoc(op);
          std::string ddPrefix = "DecisionDynamics_" + opName;

          // before_relin = arg1_degree + arg2_degree
          cstName = ddPrefix + "_0";
          MPConstraint *const ct0 =
              solver->MakeRowConstraint(0.0, 0.0, cstName);
          ct0->SetCoefficient(resultBeforeRelinVar, 1);
          if (op.getLhs() == op.getRhs()) {
            ct0->SetCoefficient(lhsDegreeVar, -2);
          } else {
            ct0->SetCoefficient(lhsDegreeVar, -1);
            ct0->SetCoefficient(rhsDegreeVar, -1);
          }

          // result_key_basis_degree >= insert_relin_op
          cstName = ddPrefix + "_1";
          MPConstraint *const ct1 =
              solver->MakeRowConstraint(0.0, inf, cstName);
          ct1->SetCoefficient(resultAfterRelinVar, 1);
          ct1->SetCoefficient(insertRelinOpDecision, -INITIAL_KEY_BASIS_DEGREE);

          // result_key_basis_degree <= 1 + (1 - insert_relin_op) * BIG_CONST
          cstName = ddPrefix + "_2";
          MPConstraint *const ct2 = solver->MakeRowConstraint(
              0.0, INITIAL_KEY_BASIS_DEGREE + IF_THEN_AUX, cstName);
          ct2->SetCoefficient(resultAfterRelinVar, 1);
          ct2->SetCoefficient(insertRelinOpDecision, IF_THEN_AUX);

          // result_key_basis_degree >= (arg1_degree + arg2_degree)
          // - insert_relin_op * BIG_CONST
          cstName = ddPrefix + "_3";
          MPConstraint *const ct3 =
              solver->MakeRowConstraint(0.0, inf, cstName);
          ct3->SetCoefficient(resultAfterRelinVar, 1);
          ct3->SetCoefficient(insertRelinOpDecision, IF_THEN_AUX);
          ct3->SetCoefficient(resultBeforeRelinVar, -1);

          // result_key_basis_degree <= (arg1_degree + arg2_degree)
          // + insert_relin_op * BIG_CONST
          cstName = ddPrefix + "_4";
          MPConstraint *const ct4 =
              solver->MakeRowConstraint(-inf, 0.0, cstName);
          ct4->SetCoefficient(resultAfterRelinVar, 1);
          ct4->SetCoefficient(insertRelinOpDecision, -IF_THEN_AUX);
          ct4->SetCoefficient(resultBeforeRelinVar, -1);
        })
        .Default([&](Operation &op) {
          // For any other op, the key basis does not change, unless we insert
          // a relin op. Because the verifier ensures the operands and results
          // have identical key bases, we can just pass through the first
          // argument.
          //
          // before_relin = arg1_degree
          // result_degree = before_relin (1 - insert_relin_op)
          //   + 1 * insert_relin_op,
          //
          // linearized due to the quadratic term input_key_basis_degree *
          // insert_relin_op

          auto inf = solver->infinity();
          if (!hasCiphertextType(op.getOperands()) ||
              !hasCiphertextType(op.getResults())) {
            return;
          }
          auto *argDegreeVar = keyBasisVars.lookup(op.getOperand(0));

          for (Value result : op.getResults()) {
            auto *resultBeforeRelinVar = beforeRelinVars.lookup(result);
            auto *resultAfterRelinVar = keyBasisVars.lookup(result);
            auto *insertRelinOpDecision = decisionVariables.lookup(&op);
            std::string opName = nameAndLoc(&op);
            std::string ddPrefix = "DecisionDynamics_" + opName;

            cstName = ddPrefix + "_0";
            // This is mildly wasteful, but the presolve will prune it out and
            // it shouldn't affect the solve time. It simply helps us do
            // bookkeeping for the before/after relin vars uniformly across
            // all cases.
            MPConstraint *const ct0 =
                solver->MakeRowConstraint(0.0, 0.0, cstName);
            ct0->SetCoefficient(resultBeforeRelinVar, 1);
            ct0->SetCoefficient(argDegreeVar, -1);

            // result_key_basis_degree >= insert_relin_op
            cstName = ddPrefix + "_1";
            MPConstraint *const ct1 =
                solver->MakeRowConstraint(0.0, inf, cstName);
            ct1->SetCoefficient(resultAfterRelinVar, 1);
            ct1->SetCoefficient(insertRelinOpDecision,
                                -INITIAL_KEY_BASIS_DEGREE);

            // result_key_basis_degree <= 1 + (1 - insert_relin_op) *
            // BIG_CONST
            cstName = ddPrefix + "_2";
            MPConstraint *const ct2 = solver->MakeRowConstraint(
                0.0, INITIAL_KEY_BASIS_DEGREE + IF_THEN_AUX, cstName);
            ct2->SetCoefficient(resultAfterRelinVar, 1);
            ct2->SetCoefficient(insertRelinOpDecision, IF_THEN_AUX);

            // result_key_basis_degree >= before_degree - insert_relin_op *
            // BIG_CONST
            cstName = ddPrefix + "_3";
            MPConstraint *const ct3 =
                solver->MakeRowConstraint(0.0, inf, cstName);
            ct3->SetCoefficient(resultAfterRelinVar, 1);
            ct3->SetCoefficient(insertRelinOpDecision, IF_THEN_AUX);
            ct3->SetCoefficient(resultBeforeRelinVar, -1);

            // result_key_basis_degree <= before_degree + insert_relin_op *
            // BIG_CONST
            cstName = ddPrefix + "_4";
            MPConstraint *const ct4 =
                solver->MakeRowConstraint(-inf, 0.0, cstName);
            ct4->SetCoefficient(resultAfterRelinVar, 1);
            ct4->SetCoefficient(insertRelinOpDecision, -IF_THEN_AUX);
            ct4->SetCoefficient(resultBeforeRelinVar, -1);
          }
        });
  });

  // Dump the model
  LLVM_DEBUG({
    std::string model_str;
    solver->ExportModelAsLpFormat(false, &model_str);
    llvm::dbgs() << model_str << "\n";
  });
  // Uncomment for debug info about model solution progress.
  // solver->EnableOutput();

  operations_research::MPSolver::ResultStatus status = solver->Solve();
  if (status != MPSolver::OPTIMAL && status != MPSolver::FEASIBLE) {
    llvm::errs() << "The problem does not have a feasible solution. Status is: "
                 << status << "\n";
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Problem solved in " << solver->wall_time()
                 << " milliseconds" << "\n";
    llvm::dbgs() << "Solution:\n";
    llvm::dbgs() << "Objective value = " << objective->Value() << "\n";
    for (auto var : solver->variables()) {
      llvm::dbgs() << var->name() << " = " << var->solution_value() << "\n";
    }
  });
  for (auto item : decisionVariables) {
    solution.insert(std::make_pair(item.first, item.second->solution_value()));
  }
  for (auto item : beforeRelinVars) {
    solutionKeyBasisDegreeBeforeRelin.insert(
        std::make_pair(item.first, (int)item.second->solution_value()));
  }

  return success();
}
}  // namespace heir
}  // namespace mlir
