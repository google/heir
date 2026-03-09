#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"        // from @com_google_absl
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project
#include "ortools/math_opt/cpp/math_opt.h"  // from @com_google_ortools

// The level describes the remaining multiplicative depth of a ciphertext.
// Bootstrap operations reset the level to a maximum value (or a target level
// if specified), allowing further operations to be performed.
//
// This implementation tracks the level state through the computation
// and uses ILP to determine optimal bootstrap placement to minimize costs
// while ensuring level constraints are satisfied.
//
// ILP Formulation:
// - Variables: level[value] for each SSA value, input_level[op] for each op,
//   bootstrap[op] for each operation
// - Constraints:
//   * Level bounds: 0 <= level[value] <= bootstrapWaterline (levels
//   0..bootstrapWaterline)
//   * Operand matching: for each op, operand_i_level == input_level[op]
//   * Multiplication: output_level = input_level - 1
//   * Non-mult: output_level = input_level
//   * Bootstrap (big-M): if bootstrap[op] = 1, level_after = bootstrapWaterline
//                        else level_after = level_before
// - Objective: minimize sum of bootstrap decisions

namespace math_opt = ::operations_research::math_opt;

#define DEBUG_TYPE "ilp-bootstrap-placement"

namespace mlir {
namespace heir {

// Helper to get secret operands
static void getSecretOperands(Operation* op, DataFlowSolver* solver,
                              SmallVector<Value>& secretOperands) {
  for (OpOperand& operand : op->getOpOperands()) {
    if (isSecret(operand.get(), solver)) {
      secretOperands.push_back(operand.get());
    }
  }
}

LogicalResult ILPBootstrapPlacementAnalysis::solve() {
  math_opt::Model model("ILPBootstrapPlacementAnalysis");

  // Get the secret.generic operation
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) {
    return failure();
  }

  Block* body = genericOp.getBody();

  int nextOpaqueId = 0;
  llvm::DenseMap<Operation*, int> opaqueIds;
  auto uniqueName = [&](Operation* op) {
    std::string varName;
    llvm::raw_string_ostream ss(varName);
    ss << op->getName().getStringRef() << "_";
    if (opaqueIds.count(op) == 0)
      opaqueIds.insert(std::make_pair(op, nextOpaqueId++));
    ss << opaqueIds.lookup(op);
    return ss.str();
  };

  // Map operations to bootstrap decision variables
  llvm::DenseMap<Operation*, math_opt::Variable> decisionVariables;
  // Map SSA values to level variables (after bootstrap decision)
  llvm::DenseMap<Value, math_opt::Variable> levelVars;
  // Map SSA values to level variables before bootstrap
  llvm::DenseMap<Value, math_opt::Variable> beforeBootstrapVars;
  // Input level: level at which operands are consumed
  llvm::DenseMap<Operation*, math_opt::Variable> inputLevelVars;

  // Big-M constant for big-M method
  // NOTE: This BIG_M does not account for "freshly encrypted" ciphertexts
  // starting at a higher level than the bootstrap waterline. This should
  // be addressed in future work.
  const int BIG_M = bootstrapWaterline;

  // Create variables for all SSA values in the body
  // First, handle block arguments (inputs)
  for (BlockArgument arg : body->getArguments()) {
    if (!isSecret(arg, solver)) continue;

    std::stringstream ss;
    ss << "levelArg" << arg.getArgNumber();
    auto levelVar =
        model.AddContinuousVariable(0, bootstrapWaterline, ss.str());
    levelVars.insert(std::make_pair(arg, levelVar));
    // Inputs start at maximum level (bootstrapWaterline)
    model.AddLinearConstraint(
        levelVar == bootstrapWaterline,
        "initLevelArg" + std::to_string(arg.getArgNumber()));
  }

  // Walk operations in the body
  for (Operation& op : body->getOperations()) {
    // Check if this operation produces secret results
    if (llvm::none_of(op.getResults(), [&](OpResult result) {
          return isSecret(result, solver);
        })) {
      continue;
    }
    if (isa<secret::YieldOp>(op)) continue;

    std::string opName = uniqueName(&op);

    // Create bootstrap decision variable for this operation
    auto bootstrapVar = model.AddBinaryVariable("bootstrap" + opName);
    decisionVariables.insert(std::make_pair(&op, bootstrapVar));

    // Create input level variable: level at which operands are consumed
    auto inputLevelVar = model.AddContinuousVariable(0, bootstrapWaterline,
                                                     "inputLevel" + opName);
    inputLevelVars.insert(std::make_pair(&op, inputLevelVar));

    // Create level variables for results
    for (OpResult result : op.getResults()) {
      if (!isSecret(result, solver)) continue;

      std::stringstream ss;
      ss << "level" << opName << result.getResultNumber();
      auto levelVar =
          model.AddContinuousVariable(0, bootstrapWaterline, ss.str());
      levelVars.insert(std::make_pair(result, levelVar));

      std::stringstream ss2;
      ss2 << "levelBefore" << opName << result.getResultNumber();
      auto beforeVar =
          model.AddContinuousVariable(0, bootstrapWaterline, ss2.str());
      beforeBootstrapVars.insert(std::make_pair(result, beforeVar));
    }
  }

  // Add constraints for operations
  for (auto& [op, _] : opaqueIds) {
    std::string opName = uniqueName(op);

    // Get secret operands
    SmallVector<Value> secretOperands;
    getSecretOperands(op, solver, secretOperands);

    // Operand matching: all operands must be at the same level when consumed.
    // inputLevel = level at which operands are consumed.
    auto inputLevelVar = inputLevelVars.at(op);
    for (size_t i = 0; i < secretOperands.size(); ++i) {
      Value operand = secretOperands[i];
      if (!levelVars.contains(operand)) continue;

      std::stringstream ss;
      ss << "operandMatch" << opName << "Op" << i;
      model.AddLinearConstraint(levelVars.at(operand) == inputLevelVar,
                                ss.str());
    }

    // Output level: for multiplication, output = input - 1; else output = input
    if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
      for (OpResult result : op->getResults()) {
        if (!isSecret(result, solver)) continue;
        if (!beforeBootstrapVars.contains(result)) continue;

        auto resultBeforeVar = beforeBootstrapVars.at(result);
        // outputLevel = inputLevel - 1
        std::stringstream ss;
        ss << "mulOutput" << opName << result.getResultNumber();
        model.AddLinearConstraint(resultBeforeVar == inputLevelVar - 1,
                                  ss.str());
        std::stringstream ssMin;
        ssMin << "mulLevelMin" << opName << result.getResultNumber();
        model.AddLinearConstraint(resultBeforeVar >= 0, ssMin.str());
      }
    } else {
      for (OpResult result : op->getResults()) {
        if (!isSecret(result, solver)) continue;
        if (!beforeBootstrapVars.contains(result)) continue;

        auto resultBeforeVar = beforeBootstrapVars.at(result);
        // outputLevel = inputLevel (level flows through unchanged)
        std::stringstream ss;
        ss << "flowOutput" << opName << result.getResultNumber();
        model.AddLinearConstraint(resultBeforeVar == inputLevelVar, ss.str());
      }
    }

    // Add bootstrap constraints using big-M method
    for (OpResult result : op->getResults()) {
      if (!isSecret(result, solver)) continue;
      if (!levelVars.contains(result) || !beforeBootstrapVars.contains(result))
        continue;

      auto resultLevelVar = levelVars.at(result);
      auto resultBeforeVar = beforeBootstrapVars.at(result);
      auto bootstrapVar = decisionVariables.at(op);

      std::stringstream ss;
      ss << "bootstrapOutput" << opName << result.getResultNumber();

      // If bootstrap = 1: level_after = bootstrapWaterline
      // If bootstrap = 0: levelAfter = levelBefore
      // Using big-M:
      // levelAfter <= bootstrapWaterline + BIG_M * (1 - bootstrap)
      // levelAfter >= bootstrapWaterline - BIG_M * (1 - bootstrap)
      // levelAfter <= levelBefore + BIG_M * bootstrap
      // levelAfter >= levelBefore - BIG_M * bootstrap

      std::string cstName1 = ss.str() + "_1";
      model.AddLinearConstraint(
          resultLevelVar <= bootstrapWaterline + BIG_M * (1 - bootstrapVar),
          cstName1);

      std::string cstName2 = ss.str() + "_2";
      model.AddLinearConstraint(
          resultLevelVar >= bootstrapWaterline - BIG_M * (1 - bootstrapVar),
          cstName2);

      std::string cstName3 = ss.str() + "_3";
      model.AddLinearConstraint(
          resultLevelVar <= resultBeforeVar + BIG_M * bootstrapVar, cstName3);

      std::string cstName4 = ss.str() + "_4";
      model.AddLinearConstraint(
          resultLevelVar >= resultBeforeVar - BIG_M * bootstrapVar, cstName4);
    }
  }

  // Objective: minimize number of bootstraps
  math_opt::LinearExpression obj;
  for (auto& [op, decisionVar] : decisionVariables) {
    obj += decisionVar;
  }
  model.Minimize(obj);

  LLVM_DEBUG({
    std::stringstream ss;
    ss << model;
    llvm::dbgs() << "--- ILP model ---\n" << ss.str() << "--- end model ---\n";
  });

  // Solve the ILP
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
      break;
    default:
      llvm::errs() << "The problem does not have a feasible solution. "
                      "Termination status code: "
                   << (int)result.termination.reason << "\n";
      return failure();
  }

  // Extract solution
  auto varMap = result.variable_values();
  for (auto& [op, decisionVar] : decisionVariables) {
    solution.insert(std::make_pair(op, varMap[decisionVar] > 0.5));
  }

  for (auto& [value, beforeVar] : beforeBootstrapVars) {
    solutionLevelBeforeBootstrap.insert(
        std::make_pair(value, (int)std::round(varMap[beforeVar])));
  }
  for (auto& [value, levelVar] : levelVars) {
    solutionLevelAfterBootstrap.insert(
        std::make_pair(value, (int)std::round(varMap[levelVar])));
  }

  return success();
}

llvm::SmallVector<Value, 32>
ILPBootstrapPlacementAnalysis::getValuesToBootstrap() const {
  llvm::SmallVector<Value, 32> out;
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp || !solver) return out;
  for (const auto& [op, insert] : solution) {
    if (!insert) continue;
    for (OpResult result : op->getResults()) {
      if (isSecret(result, solver)) out.push_back(result);
    }
  }
  return out;
}

void ILPBootstrapPlacementAnalysis::printSolution(llvm::raw_ostream& os) const {
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) {
    os << "(not a secret.generic)\n";
    return;
  }
  Block* body = genericOp.getBody();
  if (!body) return;

  os << "--- ILP bootstrap placement solution ---\n";
  os << "bootstrap waterline: " << bootstrapWaterline << "\n\n";

  // Block arguments (inputs): print as in IR, level after only
  for (BlockArgument arg : body->getArguments()) {
    auto it = solutionLevelAfterBootstrap.find(arg);
    if (it != solutionLevelAfterBootstrap.end()) {
      os << "  ";
      arg.print(os);
      os << "  level(after): " << it->second << "\n";
    }
  }

  // Ops in body order: print op and operands (as in IR), then bootstrap/levels
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(op)) continue;
    bool insertBootstrap = solution.lookup(&op);
    os << "  ";
    op.print(os);
    os << "  bootstrap=" << (insertBootstrap ? "yes" : "no");
    for (OpResult result : op.getResults()) {
      auto beforeIt = solutionLevelBeforeBootstrap.find(result);
      auto afterIt = solutionLevelAfterBootstrap.find(result);
      if (beforeIt != solutionLevelBeforeBootstrap.end() ||
          afterIt != solutionLevelAfterBootstrap.end()) {
        os << " level(before=";
        if (beforeIt != solutionLevelBeforeBootstrap.end())
          os << beforeIt->second;
        else
          os << "?";
        os << " after=";
        if (afterIt != solutionLevelAfterBootstrap.end())
          os << afterIt->second;
        else
          os << "?";
        os << ")";
      }
    }
    os << "\n";
  }
  os << "--- end solution ---\n";
}

}  // namespace heir
}  // namespace mlir
