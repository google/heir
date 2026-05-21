#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"         // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
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
// - Variables: level[value] and scale[value] for each SSA value,
//   input_level[op] and input_scale[op] for each op, bootstrap[op] for each
//   operation
// - Constraints:
//   * Level bounds: 0 <= level[value] <= bootstrapWaterline (levels
//   0..bootstrapWaterline)
//   * Operand matching: for each op, operand_i_level == input_level[op]
//   * Multiplication: output_level = input_level - 1
//   * Non-mult: output_level = input_level
//   * Scale constraints: mul input scale is the sum of operand scales,
//     rescale decisions lower scale by scaleFactorBits, and bootstrap
//     decisions require feasible level/scale input pairs.
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
  // Map SSA values to scale variables (after bootstrap decision)
  llvm::DenseMap<Value, math_opt::Variable> scaleVars;
  // Map SSA values to scale variables before bootstrap/output reduction
  llvm::DenseMap<Value, math_opt::Variable> beforeBootstrapScaleVars;
  // Input level: level at which operands are consumed
  llvm::DenseMap<Operation*, math_opt::Variable> inputLevelVars;
  // Input scale: scale at which operands are consumed.
  llvm::DenseMap<Operation*, math_opt::Variable> inputScaleVars;
  llvm::DenseMap<OpOperand*, math_opt::Variable> operandDropVars;
  llvm::DenseMap<Value, math_opt::Variable> outputDropVars;
  SmallVector<Operation*> trackedOps;

  // Big-M constant for big-M method
  // NOTE: This bigM does not account for "freshly encrypted" ciphertexts
  // starting at a higher level than the bootstrap waterline. This should
  // be addressed in future work.
  const int bigM = bootstrapWaterline;
  const int scaleMax = scaleFactorBits + 2 * scaleWaterline;
  const int inputScaleMax = 2 * scaleMax;
  const int scaleBigM = scaleMax + scaleFactorBits * bootstrapWaterline;

  // Create variables for all SSA values in the body
  // First, handle block arguments (inputs)
  for (BlockArgument arg : body->getArguments()) {
    if (!isSecret(arg, solver)) continue;

    std::stringstream ss;
    ss << "levelArg" << arg.getArgNumber();
    auto levelVar = model.AddIntegerVariable(0, bootstrapWaterline, ss.str());
    levelVars.insert(std::make_pair(arg, levelVar));
    std::stringstream ssScale;
    ssScale << "scaleArg" << arg.getArgNumber();
    auto scaleVar =
        model.AddContinuousVariable(scaleWaterline, scaleMax, ssScale.str());
    scaleVars.insert(std::make_pair(arg, scaleVar));
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

    trackedOps.push_back(&op);
    std::string opName = uniqueName(&op);

    // Create bootstrap decision variable for this operation
    auto bootstrapVar = model.AddBinaryVariable("bootstrap" + opName);
    decisionVariables.insert(std::make_pair(&op, bootstrapVar));

    // Create input level variable: level at which operands are consumed
    auto inputLevelVar =
        model.AddIntegerVariable(0, bootstrapWaterline, "inputLevel" + opName);
    inputLevelVars.insert(std::make_pair(&op, inputLevelVar));
    auto inputScaleVar = model.AddContinuousVariable(
        scaleWaterline, inputScaleMax, "inputScale" + opName);
    inputScaleVars.insert(std::make_pair(&op, inputScaleVar));

    // Create level variables for results
    for (OpResult result : op.getResults()) {
      if (!isSecret(result, solver)) continue;

      std::stringstream ss;
      ss << "level" << opName << result.getResultNumber();
      auto levelVar = model.AddIntegerVariable(0, bootstrapWaterline, ss.str());
      levelVars.insert(std::make_pair(result, levelVar));

      std::stringstream ss2;
      ss2 << "levelBefore" << opName << result.getResultNumber();
      auto beforeVar =
          model.AddIntegerVariable(0, bootstrapWaterline, ss2.str());
      beforeBootstrapVars.insert(std::make_pair(result, beforeVar));

      std::stringstream ssScale;
      ssScale << "scale" << opName << result.getResultNumber();
      auto scaleVar =
          model.AddContinuousVariable(scaleWaterline, scaleMax, ssScale.str());
      scaleVars.insert(std::make_pair(result, scaleVar));

      std::stringstream ssBeforeScale;
      ssBeforeScale << "scaleBefore" << opName << result.getResultNumber();
      auto beforeScaleVar = model.AddContinuousVariable(
          scaleWaterline, scaleMax, ssBeforeScale.str());
      beforeBootstrapScaleVars.insert(std::make_pair(result, beforeScaleVar));

      std::stringstream ss3;
      ss3 << "outputDrop" << opName << result.getResultNumber();
      auto outputDropVar =
          model.AddIntegerVariable(0, bootstrapWaterline, ss3.str());
      outputDropVars.insert(std::make_pair(result, outputDropVar));
    }
  }

  // Add constraints for operations
  for (auto& [op, _] : opaqueIds) {
    std::string opName = uniqueName(op);

    // Operand matching: all operands must be consumed at the same level.
    // Unlike the original model, a producer may stay at a higher level and a
    // specific edge/use can level-reduce into this operation.
    auto inputLevelVar = inputLevelVars.at(op);
    auto inputScaleVar = inputScaleVars.at(op);
    SmallVector<math_opt::Variable> secretOperandScaleVars;
    for (OpOperand& operandUse : op->getOpOperands()) {
      Value operand = operandUse.get();
      if (!isSecret(operand, solver)) continue;
      if (!levelVars.contains(operand)) continue;

      std::stringstream ss;
      ss << "operandLevel" << opName << "Op" << operandUse.getOperandNumber();
      auto operandLevelVar =
          model.AddIntegerVariable(0, bootstrapWaterline, ss.str());
      std::stringstream ssDrop;
      ssDrop << "operandDrop" << opName << "Op"
             << operandUse.getOperandNumber();
      auto dropVar =
          model.AddIntegerVariable(0, bootstrapWaterline, ssDrop.str());
      operandDropVars.insert(std::make_pair(&operandUse, dropVar));

      std::stringstream ssScale;
      ssScale << "operandScale" << opName << "Op"
              << operandUse.getOperandNumber();
      auto operandScaleVar =
          model.AddContinuousVariable(scaleWaterline, scaleMax, ssScale.str());
      secretOperandScaleVars.push_back(operandScaleVar);

      model.AddLinearConstraint(operandLevelVar == inputLevelVar,
                                ss.str() + "MatchInput");
      model.AddLinearConstraint(
          levelVars.at(operand) == operandLevelVar + dropVar,
          ssDrop.str() + "FromSource");
      model.AddLinearConstraint(
          operandScaleVar >= scaleVars.at(operand) - scaleFactorBits * dropVar,
          ssScale.str() + "FromSource");
      model.AddLinearConstraint(
          operandScaleVar <= scaleVars.at(operand) - scaleFactorBits * dropVar,
          ssScale.str() + "FromSourceUpper");
    }

    if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
      math_opt::LinearExpression mulInputScale;
      for (auto operandScaleVar : secretOperandScaleVars) {
        mulInputScale += operandScaleVar;
      }
      if (!secretOperandScaleVars.empty()) {
        model.AddLinearConstraint(inputScaleVar == mulInputScale,
                                  "mulInputScale" + opName);
      }
    } else {
      for (auto [i, operandScaleVar] :
           llvm::enumerate(secretOperandScaleVars)) {
        model.AddLinearConstraint(
            operandScaleVar == inputScaleVar,
            "flowInputScale" + opName + std::to_string(i));
      }
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

        auto beforeScaleVar = beforeBootstrapScaleVars.at(result);
        std::stringstream ssScale;
        ssScale << "mulScaleOutput" << opName << result.getResultNumber();
        // The pass always materializes mgmt.modreduce after multiplication;
        // in CKKS this is the ordinary post-mul rescale by scaleFactorBits.
        model.AddLinearConstraint(
            beforeScaleVar == inputScaleVar - scaleFactorBits, ssScale.str());
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

        auto beforeScaleVar = beforeBootstrapScaleVars.at(result);
        std::stringstream ssScale;
        ssScale << "flowScaleOutput" << opName << result.getResultNumber();
        model.AddLinearConstraint(beforeScaleVar == inputScaleVar,
                                  ssScale.str());
      }
    }

    // Add bootstrap constraints using big-M method
    for (OpResult result : op->getResults()) {
      if (!isSecret(result, solver)) continue;
      if (!levelVars.contains(result) || !beforeBootstrapVars.contains(result))
        continue;

      auto resultLevelVar = levelVars.at(result);
      auto resultBeforeVar = beforeBootstrapVars.at(result);
      auto resultScaleVar = scaleVars.at(result);
      auto resultBeforeScaleVar = beforeBootstrapScaleVars.at(result);
      auto outputDropVar = outputDropVars.at(result);
      auto bootstrapVar = decisionVariables.at(op);

      std::stringstream ss;
      ss << "bootstrapOutput" << opName << result.getResultNumber();

      // If bootstrap = 1: level_after = bootstrapWaterline
      // If bootstrap = 0: levelAfter = levelBefore - outputDrop
      // Using big-M:
      // levelAfter <= bootstrapWaterline + bigM * (1 - bootstrap)
      // levelAfter >= bootstrapWaterline - bigM * (1 - bootstrap)
      // levelAfter <= levelBefore - outputDrop + bigM * bootstrap
      // levelAfter >= levelBefore - outputDrop - bigM * bootstrap

      std::string cstName1 = ss.str() + "_1";
      model.AddLinearConstraint(
          resultLevelVar <= bootstrapWaterline + bigM * (1 - bootstrapVar),
          cstName1);

      std::string cstName2 = ss.str() + "_2";
      model.AddLinearConstraint(
          resultLevelVar >= bootstrapWaterline - bigM * (1 - bootstrapVar),
          cstName2);

      std::string cstName3 = ss.str() + "_3";
      model.AddLinearConstraint(resultLevelVar <= resultBeforeVar -
                                                      outputDropVar +
                                                      bigM * bootstrapVar,
                                cstName3);

      std::string cstName4 = ss.str() + "_4";
      model.AddLinearConstraint(resultLevelVar >= resultBeforeVar -
                                                      outputDropVar -
                                                      bigM * bootstrapVar,
                                cstName4);

      // Scale-aware bootstrap feasibility: bootstrapping is only
      // valid when the input scale fits the input level, and a bootstrap
      // produces a ciphertext with at least the base scale. Without
      // bootstrapping, output rescale decisions lower scale by scaleFactorBits
      // per dropped level.
      std::string scaleCstName1 = ss.str() + "_scale_bts_input";
      model.AddLinearConstraint(
          resultBeforeScaleVar <=
              scaleFactorBits *
                      (resultBeforeVar - bootstrapLevelLowerBound + 1) +
                  scaleBigM * (1 - bootstrapVar),
          scaleCstName1);

      std::string scaleCstName2 = ss.str() + "_scale_bts_output";
      model.AddLinearConstraint(
          resultScaleVar >= scaleFactorBits - scaleBigM * (1 - bootstrapVar),
          scaleCstName2);

      std::string scaleCstName3 = ss.str() + "_scale_nobts_output";
      model.AddLinearConstraint(
          resultScaleVar >= resultBeforeScaleVar -
                                scaleFactorBits * outputDropVar -
                                scaleBigM * bootstrapVar,
          scaleCstName3);

      std::string scaleCstName4 = ss.str() + "_scale_nobts_output_upper";
      model.AddLinearConstraint(
          resultScaleVar <= resultBeforeScaleVar -
                                scaleFactorBits * outputDropVar +
                                scaleBigM * bootstrapVar,
          scaleCstName4);
    }
  }

  if (useOrbitCompression) {
    llvm::DenseMap<Operation*, std::string> opColors;
    llvm::DenseMap<Value, std::string> valueColors;

    for (BlockArgument arg : body->getArguments()) {
      if (isSecret(arg, solver)) {
        valueColors.insert(
            std::make_pair(arg, ("arg" + std::to_string(arg.getArgNumber()))));
      }
    }
    for (Operation* op : trackedOps) {
      opColors.insert(std::make_pair(op, op->getName().getStringRef().str()));
      for (OpResult result : op->getResults()) {
        if (isSecret(result, solver)) valueColors[result] = opColors.lookup(op);
      }
    }

    auto join = [](SmallVector<std::string>& parts) {
      std::sort(parts.begin(), parts.end());
      std::string out;
      llvm::raw_string_ostream os(out);
      for (const auto& part : parts) os << part << ";";
      return os.str();
    };

    int previousGroupCount = -1;
    for (int iter = 0; iter < 32; ++iter) {
      std::map<std::string, int> descriptorToGroup;
      llvm::DenseMap<Operation*, std::string> nextOpColors;
      llvm::DenseMap<Value, std::string> nextValueColors = valueColors;

      for (Operation* op : trackedOps) {
        SmallVector<Value> secretOperands;
        getSecretOperands(op, solver, secretOperands);
        SmallVector<std::string> operandColors;
        for (Value operand : secretOperands) {
          auto it = valueColors.find(operand);
          operandColors.push_back(it == valueColors.end() ? "external"
                                                          : it->second);
        }

        SmallVector<std::string> userColors;
        for (OpResult result : op->getResults()) {
          if (!isSecret(result, solver)) continue;
          for (Operation* user : result.getUsers()) {
            auto it = opColors.find(user);
            userColors.push_back(it == opColors.end() ? "external"
                                                      : it->second);
          }
        }

        std::string descriptor;
        llvm::raw_string_ostream os(descriptor);
        os << op->getName().getStringRef() << "|in=" << join(operandColors)
           << "|out=" << join(userColors) << "|results=";
        for (OpResult result : op->getResults()) {
          if (isSecret(result, solver)) os << result.getResultNumber() << ":";
        }

        auto [it, inserted] = descriptorToGroup.insert(
            std::make_pair(os.str(), descriptorToGroup.size()));
        std::string color = "orbit_group_" + std::to_string(it->second);
        nextOpColors[op] = color;
        for (OpResult result : op->getResults()) {
          if (isSecret(result, solver)) nextValueColors[result] = color;
        }
      }

      opColors = std::move(nextOpColors);
      valueColors = std::move(nextValueColors);
      int groupCount = descriptorToGroup.size();
      if (groupCount == previousGroupCount) break;
      previousGroupCount = groupCount;
    }

    std::map<std::string, SmallVector<Operation*>> groups;
    for (Operation* op : trackedOps) groups[opColors.lookup(op)].push_back(op);

    for (auto& [_, group] : groups) {
      if (group.size() <= 1) continue;
      Operation* anchor = group.front();
      for (Operation* op : llvm::drop_begin(group)) {
        model.AddLinearConstraint(
            decisionVariables.at(op) == decisionVariables.at(anchor),
            "orbitBootstrapDecision" + uniqueName(op));
        model.AddLinearConstraint(
            inputLevelVars.at(op) == inputLevelVars.at(anchor),
            "orbitInputLevel" + uniqueName(op));
        model.AddLinearConstraint(
            inputScaleVars.at(op) == inputScaleVars.at(anchor),
            "orbitInputScale" + uniqueName(op));

        for (auto [anchorResult, result] :
             llvm::zip_equal(anchor->getResults(), op->getResults())) {
          if (!isSecret(anchorResult, solver) || !isSecret(result, solver)) {
            continue;
          }
          model.AddLinearConstraint(
              levelVars.at(result) == levelVars.at(anchorResult),
              "orbitLevel" + uniqueName(op) +
                  std::to_string(result.getResultNumber()));
          model.AddLinearConstraint(
              beforeBootstrapVars.at(result) ==
                  beforeBootstrapVars.at(anchorResult),
              "orbitBeforeLevel" + uniqueName(op) +
                  std::to_string(result.getResultNumber()));
          model.AddLinearConstraint(
              scaleVars.at(result) == scaleVars.at(anchorResult),
              "orbitScale" + uniqueName(op) +
                  std::to_string(result.getResultNumber()));
          model.AddLinearConstraint(
              beforeBootstrapScaleVars.at(result) ==
                  beforeBootstrapScaleVars.at(anchorResult),
              "orbitBeforeScale" + uniqueName(op) +
                  std::to_string(result.getResultNumber()));
        }
      }
    }
  }

  // Objective: minimize a weighted placement cost.
  math_opt::LinearExpression obj;
  for (auto& [op, decisionVar] : decisionVariables) {
    obj += bootstrapCost * decisionVar;
  }
  for (auto& [value, dropVar] : outputDropVars) {
    obj += rescaleCost * dropVar;
  }
  for (auto& [operand, dropVar] : operandDropVars) {
    obj += rescaleCost * dropVar;
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
  for (auto& [operand, dropVar] : operandDropVars) {
    int levelToDrop = (int)std::round(varMap[dropVar]);
    if (levelToDrop <= 0) continue;
    operandLevelReductions.push_back(
        {operand->getOwner(), operand->getOperandNumber(), levelToDrop});
  }
  for (auto& [value, dropVar] : outputDropVars) {
    int levelToDrop = (int)std::round(varMap[dropVar]);
    if (levelToDrop <= 0) continue;
    outputLevelReductions.push_back({value, levelToDrop});
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
