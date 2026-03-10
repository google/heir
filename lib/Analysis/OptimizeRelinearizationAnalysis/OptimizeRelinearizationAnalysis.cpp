#include "lib/Analysis/OptimizeRelinearizationAnalysis/OptimizeRelinearizationAnalysis.h"

#include <cassert>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"             // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h" // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

// Avoid copybara mangling and separate third party includes with a comment.
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"        // from @com_google_absl
// OR-Tools dependency
#include "ortools/math_opt/cpp/math_opt.h"  // from @com_google_ortools

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
// For now, fix the key basis degree bound to 2 (dimension 3). Could make it a
// pass flag later, or introduce costs for high-degree operations and let it be
// unconstrained.
constexpr int MAX_KEY_BASIS_DEGREE = 2;
constexpr int IF_THEN_AUX = 100;

namespace math_opt = ::operations_research::math_opt;

namespace mlir {
namespace heir {

#define DEBUG_TYPE "optimize-relinearization-analysis"

LogicalResult OptimizeRelinearizationAnalysis::solve() {
  math_opt::Model model("OptimizeRelinearizationAnalysis");

  // If the pass option use-loc-based-variable-names is set, then the variable
  // names will use the op's Location attribute. This should only be set when
  // --optimize-relinearization is the only pass applied, as otherwise the loc
  // is not guaranteed to be unique and this analysis may fail. This is useful
  // when debugging, as a failing IR can be printed before running this pass in
  // isolation.
  int nextOpaqueId = 0;
  llvm::DenseMap<Operation*, int> opaqueIds;
  auto uniqueName = [&](Operation* op) {
    std::string varName;
    llvm::raw_string_ostream ss(varName);
    ss << op->getName() << "_";
    if (useLocBasedVariableNames) {
      ss << op->getLoc();
    } else {
      if (opaqueIds.count(op) == 0)
        opaqueIds.insert(std::make_pair(op, nextOpaqueId++));

      ss << opaqueIds.lookup(op);
    }
    return ss.str();
  };

  // Map an operation to a decision to relinearize its results.
  llvm::DenseMap<Operation*, math_opt::Variable> decisionVariables;
  // keyBasisVars maps SSA values to variables tracking the key basis degree
  // of the ciphertext at that point in the computation. If the SSA value is
  // the result of an op, this variable corresponds to the degree _after_ the
  // decision to relinearize is applied.
  llvm::DenseMap<Value, math_opt::Variable> keyBasisVars;
  // beforeRelinVars is the same as keyBasisVars, but _before_
  // the decision to relinearize is applied. We need both because the
  // post-processing of the solution requires us to remember the before-relin
  // key basis degree. We could recompute it later, but it's more general to
  // track it.
  llvm::DenseMap<Value, math_opt::Variable> beforeRelinVars;

  // First create a variable for each SSA value tracking the key basis degree
  // of the ciphertext at that point in the computation, as well as the decision
  // variable to track whether to insert a relinearization operation after the
  // operation.
  opToRunOn->walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    // Skipping loop bodies because they will be handled by a recursive solver.
    // But,we still need to create variables for the loop's results
    // so the outer solver can reason about the loop.
    if (isa<LoopLikeOpInterface>(op) && op != opToRunOn) {
      std::string name = uniqueName(op);
      if (isSecret(op->getResults(), solver)) {
        auto decisionVar = model.AddBinaryVariable("InsertRelin_" + name);
        decisionVariables.insert(std::make_pair(op, decisionVar));
      }

      for (OpResult opResult : op->getOpResults()) {
        Value result = opResult;
        if (!isSecret(result, solver)) {
          continue;
        }
        std::string varName =
            "Degree_" + name + "_" + std::to_string(opResult.getResultNumber());
        auto keyBasisVar =
            model.AddContinuousVariable(0, MAX_KEY_BASIS_DEGREE, varName);
        keyBasisVars.insert(std::make_pair(result, keyBasisVar));

        std::string brVarName = varName + "_br";
        auto brKeyBasisVar =
            model.AddContinuousVariable(0, MAX_KEY_BASIS_DEGREE, brVarName);
        beforeRelinVars.insert(std::make_pair(result, brKeyBasisVar));
      }
      return WalkResult::skip();
    }

    std::string name = uniqueName(op);

    if (isa<ModuleOp>(op)) {
      return WalkResult::advance();
    }

    // skip secret generic op; we decide inside generic op block
    if (!isa<secret::GenericOp>(op) && isSecret(op->getResults(), solver)) {
      auto decisionVar = model.AddBinaryVariable("InsertRelin_" + name);
      decisionVariables.insert(std::make_pair(op, decisionVar));
    }

    // Except for block arguments, SSA values are created as results of
    // operations. Create one keyBasisDegree variable for each op result.
    std::string varName = "Degree_" + name;
    for (OpResult opResult : op->getOpResults()) {
      // skip secret generic ops
      Value result = opResult;
      varName = varName + "_" + std::to_string(opResult.getResultNumber());

      if (isa<secret::GenericOp>(op) || !isSecret(result, solver)) {
        continue;
      }

      auto keyBasisVar =
          model.AddContinuousVariable(0, MAX_KEY_BASIS_DEGREE, varName);
      keyBasisVars.insert(std::make_pair(result, keyBasisVar));

      // br means "before relin"
      std::string brVarName = varName + "_br";
      auto brKeyBasisVar =
          model.AddContinuousVariable(0, MAX_KEY_BASIS_DEGREE, brVarName);
      beforeRelinVars.insert(std::make_pair(result, brKeyBasisVar));
    }

    // Handle block arguments to the op, which are assumed to already be
    // linearized, though this could be generalized to read the degree from the
    // type.
    if (op->getNumRegions() == 0) {
      return WalkResult::advance();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Handling block arguments for " << op->getName() << "\n");
    for (Region& region : op->getRegions()) {
      for (Block& block : region.getBlocks()) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isSecret(arg, solver)) {
            continue;
          }

          std::stringstream ss;
          ss << "Degree_ba" << arg.getArgNumber() << "_" << name;
          std::string varName = ss.str();
          auto keyBasisVar =
              model.AddContinuousVariable(0, MAX_KEY_BASIS_DEGREE, varName);
          keyBasisVars.insert(std::make_pair(arg, keyBasisVar));
        }
      }
    }
    return WalkResult::advance();
  });

  // Constraints to initialize the key basis degree variables at the start of
  // the computation.
  for (auto& [value, var] : keyBasisVars) {
    if (llvm::isa<BlockArgument>(value)) {
      // If the dimension is 3, the key basis is [0, 1, 2] and the degree is 2.
      auto constrainedDegree = getDimension(value, solver).value_or(2) - 1;
      model.AddLinearConstraint(var == constrainedDegree, "");
    }
  }

  std::string cstName;
  // For each operation, constrain its inputs to all have the same key basis
  // degree. Most FHE backends we're aware of do not require this, and can
  // handle mixed-degree operations like ciphertext addition. When we do require
  // this, the output of an operation like ciphertext addition can be passed
  // through from the input unchanged. If we don't require this, the output
  // of the addition must be a max over the input degrees.
  if (!allowMixedDegreeOperands) {
    opToRunOn->walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
      // Skip loop bodies — they will be handled by a recursive solver
      if (isa<LoopLikeOpInterface>(op) && op != opToRunOn) {
        return WalkResult::skip();
      }

      if (op->getNumOperands() <= 1) {
        return WalkResult::advance();
      }

      // secret generic op arguments are not constrained
      // instead their block arguments are constrained
      if (isa<secret::GenericOp>(op)) {
        return WalkResult::advance();
      }

      std::string name = uniqueName(op);

      // only equality for secret operands
      SmallVector<OpOperand*, 4> secretOperands;
      getSecretOperands(op, secretOperands, solver);
      if (secretOperands.size() <= 1) {
        return WalkResult::advance();
      }

      auto anchorVar = keyBasisVars.at(secretOperands[0]->get());

      // degree(operand 0) == degree(operand i)
      for (OpOperand* opOperand : secretOperands) {
        if (!keyBasisVars.contains(opOperand->get())) {
          continue;
        }
        auto operandDegreeVar = keyBasisVars.at(opOperand->get());
        if (anchorVar == operandDegreeVar) {
          continue;
        }
        std::stringstream ss;
        ss << "ArgKeyBasisEquality_" << opOperand->getOperandNumber() << "_"
           << name;
        model.AddLinearConstraint(operandDegreeVar == anchorVar, ss.str());
      }
      return WalkResult::advance();
    });
  }

  // Some ops require a linear key basis. Yield is a special case
  // where we require returned values from funcs to be linearized.
  // TODO(#1398): determine whether we need linear key basis for modreduce.
  opToRunOn->walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    if (isa<LoopLikeOpInterface>(op) && op != opToRunOn) {
      return WalkResult::skip();
    }
    llvm::TypeSwitch<Operation&>(*op)
        .Case<tensor_ext::RotateOp, secret::YieldOp, mgmt::ModReduceOp>(
            [&](auto op) {
              for (OpOperand& operand : op->getOpOperands()) {
                // skip non secret argument
                if (!isSecret(operand.get(), solver)) {
                  continue;
                }
                if (!keyBasisVars.contains(operand.get())) {
                  // This could happen if you return a block argument without
                  // doing anything to it. No variables are created, but it does
                  // not necessarily need to be constrained.
                  if (isa<secret::YieldOp>(op)) return;

                  assert(false && "Operand not found in keyBasisVars");
                }
                auto operandDegreeVar = keyBasisVars.at(operand.get());
                std::stringstream ss;
                ss << "RequireLinearized_" << uniqueName(op) << "_"
                   << operand.getOperandNumber();
                model.AddLinearConstraint(operandDegreeVar == 1, ss.str());
              }
            })
        .Case<affine::AffineYieldOp, scf::YieldOp>([&](auto op) {
          // For loop yield ops, the degree returned must not exceed the degree
          // of the corresponding iter_arg block argument at the start of the
          // loop. This prevents unbounded growth across loop iterations.
          auto parentLoop = op->getParentOp();
          auto loopLike = dyn_cast<LoopLikeOpInterface>(parentLoop);
          if (!loopLike) return;

          auto iterArgs = loopLike.getRegionIterArgs();

          // Number of iter args should match number of yielded operands
          if (iterArgs.size() != op.getNumOperands()) return;

          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            auto yieldOperand = op.getOperand(i);
            if (!isSecret(yieldOperand, solver)) continue;
            if (!keyBasisVars.contains(yieldOperand)) continue;

            auto yieldDegreeVar = keyBasisVars.at(yieldOperand);
            auto iterArgDegreeVar = keyBasisVars.at(iterArgs[i]);

            model.AddLinearConstraint(
                yieldDegreeVar <= iterArgDegreeVar,
                "LoopCarriedDependency_" + std::to_string(i) + "_" +
                    uniqueName(op));
          }
        });
    return WalkResult::advance();
  });

  // When mixed-degree ops are enabled, the default result degree of an op is
  // the max of the operand degree. This next block of code adds inequality
  // constraints to ensure the result is larger than the arguments, but it will
  // not be constrained by the model to be equal to that max value.
  std::unordered_set<const math_opt::Variable*> extraVarsForObjective;

  // Add constraints that set the before_relin variables appropriately
  opToRunOn->walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    // Skip loop bodies — they will be handled by a recursive solver
    if (isa<LoopLikeOpInterface>(op) && op != opToRunOn) {
      return WalkResult::skip();
    }

    llvm::TypeSwitch<Operation&>(*op)
        .Case<arith::MulIOp, arith::MulFOp>([&](auto op) {
          // if plain mul, skip
          if (!isSecret(op.getResult(), solver)) {
            return WalkResult::advance();
          }
          // ct-ct mul
          if (isSecret(op.getLhs(), solver) && isSecret(op.getRhs(), solver)) {
            auto lhsDegreeVar = keyBasisVars.at(op.getLhs());
            auto rhsDegreeVar = keyBasisVars.at(op.getRhs());
            auto resultBeforeRelinVar = beforeRelinVars.at(op.getResult());
            std::string opName = uniqueName(op);
            std::string ddPrefix = "BeforeRelin_" + opName;

            // before_relin = arg1_degree + arg2_degree
            cstName = ddPrefix + "_0";
            if (op.getLhs() == op.getRhs()) {
              model.AddLinearConstraint(
                  resultBeforeRelinVar == 2 * lhsDegreeVar, cstName);
            } else {
              model.AddLinearConstraint(
                  resultBeforeRelinVar == rhsDegreeVar + lhsDegreeVar, cstName);
            }
          } else {
            // ct-pt op
            SmallVector<OpOperand*, 4> secretOperands;
            getSecretOperands(op, secretOperands, solver);
            auto argDegreeVar = keyBasisVars.at(secretOperands[0]->get());

            // similar logic to the Default below
            auto resultBeforeRelinVar = beforeRelinVars.at(op.getResult());
            std::string opName = uniqueName(op);
            std::string ddPrefix = "DecisionDynamics_" + opName;

            cstName = ddPrefix + "_0";
            // This is mildly wasteful, but the presolve will prune it out and
            // it shouldn't affect the solve time. It simply helps us do
            // bookkeeping for the before/after relin vars uniformly across
            // all cases.
            model.AddLinearConstraint(resultBeforeRelinVar == argDegreeVar,
                                      cstName);
          }
        })
        .Default([&](Operation& op) {
          // Handling of for loop ops, usinf the solved inner degree
          if (isa<LoopLikeOpInterface>(op)) {
            for (auto [idx, result] : llvm::enumerate(op.getResults())) {
              if (!isSecret(result, solver)) continue;
              auto resultBeforeRelinVar = beforeRelinVars.at(result);
              // Use the degree that the inner solver computed for this yield
              int solvedDegree = loopBoundaryDegrees[&op][idx];
              model.AddLinearConstraint(
                  resultBeforeRelinVar == solvedDegree,
                  "LoopOutputDegree_" + uniqueName(&op) + "_" +
                      std::to_string(idx));
            }
            return WalkResult::advance();
          }

          // For any other op, the key basis does not change unless we insert
          // a relin op. The operands may have the same basis degree, if that
          // is required by the backend and allowMixedDegreeOperands is false,
          // in which case we can just forward the degree of the first secret
          // operand. Otherwise, we have to require the output to be the max
          // of the inputs, which requires inequality constraints.
          //
          // before_relin = arg1_degree
          //
          // or,
          // before_relin >= arg1_degree
          // before_relin >= arg2_degree
          // before_relin >= arg3_degree

          // secret generic op arguments are not constrained
          // instead their block arguments are constrained
          if (isa<secret::GenericOp>(op)) {
            return WalkResult::advance();
          }
          if (!isSecret(op.getResults(), solver)) {
            return WalkResult::advance();
          }
          SmallVector<OpOperand*, 4> secretOperands;
          getSecretOperands(&op, secretOperands, solver);

          std::string opName = uniqueName(&op);
          if (allowMixedDegreeOperands) {
            for (OpOperand* opOperand : secretOperands) {
              std::string ddPrefix =
                  "DecisionDynamics_Mixed_" + opName + "_" +
                  std::to_string(opOperand->getOperandNumber());
              auto argDegreeVar = keyBasisVars.at(opOperand->get());
              for (OpResult opResult : op.getOpResults()) {
                Value result = opResult;
                const math_opt::Variable& resultBeforeRelinVar =
                    beforeRelinVars.at(result);
                cstName =
                    ddPrefix + "_" + std::to_string(opResult.getResultNumber());
                model.AddLinearConstraint(resultBeforeRelinVar >= argDegreeVar,
                                          cstName);
                extraVarsForObjective.insert(&resultBeforeRelinVar);
              }
            }
          } else {
            auto argDegreeVar = keyBasisVars.at(secretOperands[0]->get());

            for (Value result : op.getResults()) {
              auto resultBeforeRelinVar = beforeRelinVars.at(result);
              std::string opName = uniqueName(&op);
              std::string ddPrefix = "DecisionDynamics_" + opName;

              cstName = ddPrefix + "_0";
              // This is mildly wasteful, but the presolve will prune it out and
              // it shouldn't affect the solve time. It simply helps us do
              // bookkeeping for the before/after relin vars uniformly across
              // all cases.
              model.AddLinearConstraint(resultBeforeRelinVar == argDegreeVar,
                                        cstName);
            }
          }
        });
    return WalkResult::advance();
  });

  // The objective is to minimize the number of relinearization ops.
  // TODO(#1018): improve the objective function to account for differing costs
  // of operations at varying degrees, as well as the cost of relinearizing
  // based on the starting degree of the input.
  math_opt::LinearExpression obj;
  for (auto& [op, decisionVar] : decisionVariables) {
    obj += decisionVar;
  }
  model.Minimize(obj);

  // Add constraints that control the effect of relinearization insertion.
  opToRunOn->walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    // Skip loop bodies — they will be handled by a recursive solver
    if (isa<LoopLikeOpInterface>(op) && op != opToRunOn) {
      return WalkResult::skip();
    }

    // We don't need a type switch here because the only difference
    // between mul and other ops is how the before_relin variable is related to
    // the operand variables.
    //
    // result_degree = before_relin (1 - insert_relin_op)
    //   + 1 * insert_relin_op,
    //
    // linearized due to the quadratic term before_relin * insert_relin_op

    // secret generic op arguments are not constrained
    // instead their block arguments are constrained
    if (isa<secret::GenericOp>(op)) {
      return WalkResult::advance();
    }
    if (!isSecret(op->getResults(), solver)) {
      return WalkResult::advance();
    }

    for (OpResult opResult : op->getResults()) {
      Value result = opResult;
      auto resultBeforeRelinVar = beforeRelinVars.at(result);
      auto resultAfterRelinVar = keyBasisVars.at(result);
      auto insertRelinOpDecision = decisionVariables.at(op);
      std::string opName = uniqueName(op);
      std::string ddPrefix = "DecisionDynamics_" + opName + "_" +
                             std::to_string(opResult.getResultNumber());

      cstName = ddPrefix + "_1";
      model.AddLinearConstraint(resultAfterRelinVar >= insertRelinOpDecision,
                                cstName);

      cstName = ddPrefix + "_2";
      model.AddLinearConstraint(
          resultAfterRelinVar <= 1 + IF_THEN_AUX * (1 - insertRelinOpDecision),
          cstName);

      cstName = ddPrefix + "_3";
      model.AddLinearConstraint(
          resultAfterRelinVar >=
              resultBeforeRelinVar - IF_THEN_AUX * insertRelinOpDecision,
          cstName);

      cstName = ddPrefix + "_4";
      model.AddLinearConstraint(
          resultAfterRelinVar <=
              resultBeforeRelinVar + IF_THEN_AUX * insertRelinOpDecision,
          cstName);
    }
    return WalkResult::advance();
  });

  // Dump the model
  LLVM_DEBUG({
    std::stringstream ss;
    ss << model;
    llvm::dbgs() << ss.str();
  });

  const absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip);

  if (!status.ok()) {
    std::stringstream ss;
    ss << "Error solving the problem: " << status.status() << "\n";
    llvm::errs() << ss.str();
    return failure();
  }

  const math_opt::SolveResult& result = status.value();

  switch (result.termination.reason) {
    case math_opt::TerminationReason::kOptimal:
    case math_opt::TerminationReason::kFeasible:
      LLVM_DEBUG({
        llvm::dbgs() << "Problem solved in "
                     << result.solve_time() / absl::Milliseconds(1)
                     << " milliseconds.\n"
                     << "Solution:\n";
        llvm::dbgs() << "Objective value = " << result.objective_value()
                     << "\n";
        for (const auto& [var, value] : result.variable_values()) {
          llvm::dbgs() << var.name() << " = " << value << "\n";
        }
      });
      break;
    default:
      llvm::errs() << "The problem does not have a feasible solution. "
                      "Termination status code: "
                   << (int)result.termination.reason
                   << " (see: "
                      "https://github.com/google/or-tools/blob/"
                      "ed94162b910fa58896db99191378d3b71a5313af/ortools/"
                      "math_opt/cpp/solve_result.h#L124)"
                   << "\n";
      return failure();
  }

  auto varMap = result.variable_values();
  for (auto item : decisionVariables) {
    solution.insert(std::make_pair(item.first, varMap[item.second]));
  }
  for (auto item : beforeRelinVars) {
    solutionKeyBasisDegreeBeforeRelin.insert(
        (std::make_pair(item.first, (int)varMap[item.second])));
  }

  return success();
}
}  // namespace heir
}  // namespace mlir
