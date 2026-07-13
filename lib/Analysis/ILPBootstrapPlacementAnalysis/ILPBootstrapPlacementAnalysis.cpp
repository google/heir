#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"

#include <cmath>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
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
// Bootstrap operations reset the level to a maximum value, after which decoded
// management ops may reduce it to the target level chosen by the ILP. In CKKS
// mode this analysis also tracks scale bits:
//   Sw: waterline/base scale bits (scaleWaterline)
//   Sf: scale bits dropped by one rescale/modreduce (scaleFactorBits)
//
// This implementation tracks level/scale state through one secret.generic body
// and uses ILP to determine cost-minimal bootstrap and rescale placement while
// keeping producer values, operand edges, and op results mutually feasible.
// A later transform decodes the solution into mgmt operations.
//
// ILP formulation:
// - Variables:
//   * level[value], scale[value] (in bits) for each secret SSA value
//   * input_level[op], input_scale[op] for each tracked op
//   * bootstrap[op], node_rescale[op] for management after an op
//   * edge_rescale[use] and, for CKKS multiplication operands, edge_scale[use]
//     for management before a consuming op
// - Initialization:
//   * secret.generic body args are initialized from associated mgmt.mgmt attrs
//     when present, otherwise from (bootstrapWaterline, Sw)
// - Constraints:
//   * bounds: levels are 0..bootstrapWaterline; CKKS scales are Sw..scaleMax
//   * level-only mode fixes all live scales to Sw
//   * operand edges allow level/scale reduction before the consuming op
//   * non-multiplication operands share the consumer's input_scale
//   * CKKS multiplication input_scale is the sum of operand edge scales;
//     plaintext constants contribute Sw
//   * node transitions relate each op's input state to each result state by
//     either direct rescale/modswitch management or a bootstrap transition
//   * a yielded value is pinned to the level annotated on its secret.generic
//     result by an mgmt.mgmt attr, and to the annotated scale when nonzero
// - Objective: minimize total bootstrap and rescale cost. With a per-level cost
//   model, also charge each tracked op its latency at its input level. A tiny
//   per-level term breaks ties toward higher levels for output values.

namespace math_opt = ::operations_research::math_opt;

#define DEBUG_TYPE "ilp-bootstrap-placement"

namespace mlir {
namespace heir {

using ScaleMode = ILPBootstrapPlacementAnalysis::ScaleMode;

static bool isMultiplication(Operation* op) {
  return isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op);
}

static bool isAdditionLike(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op);
}

static bool isConstantLike(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() != nullptr;
}

// The level-dependent latency term for one op, or nullopt if the op class has
// no level-dependent cost.
static std::optional<LinearCost> levelCostForOp(Operation* op,
                                                DataFlowSolver* solver,
                                                const OpCostModel& costModel) {
  bool ctCt = llvm::count_if(op->getOperands(), [&](auto opd) {
                return isSecret(opd, solver);
              }) >= 2;
  if (isMultiplication(op)) return ctCt ? costModel.mulCtCt : costModel.mulCtPt;
  if (isAdditionLike(op)) return ctCt ? costModel.addCtCt : costModel.addCtPt;
  if (isa<arith::NegFOp>(op)) return costModel.negate;
  if (isa<tensor_ext::RotateOp>(op)) return costModel.rotate;
  return std::nullopt;
}

static int roundedValue(const math_opt::VariableMap<double>& varMap,
                        const math_opt::Variable& var) {
  return static_cast<int>(std::round(varMap.at(var)));
}

struct ILPModelState {
  ILPModelState(Block* body, DataFlowSolver* solver, int bootstrapWaterline,
                int scaleWaterline, int scaleFactorBits, ScaleMode scaleMode)
      : model("ILPBootstrapPlacementAnalysis"),
        body(body),
        solver(solver),
        bootstrapWaterline(bootstrapWaterline),
        levelOnly(scaleMode == ScaleMode::kLevelOnly),
        sw(scaleWaterline),
        sf(scaleFactorBits),
        scaleMax(levelOnly ? sw : sf + 2 * sw),
        bigM(bootstrapWaterline + 1),
        scaleBigM(4 * scaleMax + sf * (bootstrapWaterline + 1)) {}

  math_opt::Variable addLevelVar(int lower, int upper,
                                 const std::string& name) {
    return model.AddIntegerVariable(lower, upper, name);
  }

  math_opt::Variable addValueLevelVar(int lower, int upper,
                                      const std::string& name) {
    return model.AddContinuousVariable(lower, upper, name);
  }

  math_opt::Variable addScaleVar(const std::string& name) {
    return model.AddIntegerVariable(sw, scaleMax, name);
  }

  std::string uniqueName(Operation* op) {
    std::string varName;
    llvm::raw_string_ostream ss(varName);
    ss << op->getName().getStringRef() << "_";
    if (opaqueIds.count(op) == 0)
      opaqueIds.insert(std::make_pair(op, nextOpaqueId++));
    ss << opaqueIds.lookup(op);
    return ss.str();
  }

  math_opt::Model model;
  Block* body;
  DataFlowSolver* solver;
  int bootstrapWaterline;
  bool levelOnly;
  int sw;
  int sf;
  int scaleMax;
  int bigM;
  int scaleBigM;
  int nextOpaqueId = 0;
  llvm::DenseMap<Operation*, int> opaqueIds;
  llvm::DenseMap<Value, math_opt::Variable> valueLevelVars;
  llvm::DenseMap<Value, math_opt::Variable> valueScaleVars;
  llvm::DenseMap<Operation*, math_opt::Variable> inputLevelVars;
  llvm::DenseMap<Operation*, math_opt::Variable> inputScaleVars;
  llvm::DenseMap<Operation*, math_opt::Variable> nodeRescaleVars;
  llvm::DenseMap<Operation*, math_opt::Variable> bootstrapVars;
  llvm::DenseMap<OpOperand*, math_opt::Variable> edgeScaleVars;
  llvm::DenseMap<OpOperand*, math_opt::Variable> edgeRescaleVars;
  SmallVector<Operation*> trackedOps;
};

static LogicalResult addBodyArgumentVariables(ILPModelState& state) {
  for (BlockArgument arg : state.body->getArguments()) {
    if (!isSecret(arg, state.solver)) continue;

    int initialLevel = state.bootstrapWaterline;
    int initialScale = state.sw;
    if (mgmt::MgmtAttr mgmtAttr = mgmt::findMgmtAttrAssociatedWith(arg)) {
      initialLevel = mgmtAttr.getLevel();
      if (!state.levelOnly && mgmtAttr.getScale() != 0) {
        initialScale = mgmtAttr.getScale();
      }
    }

    Operation* parentOp = state.body->getParentOp();
    if (initialLevel < 0 || initialLevel > state.bootstrapWaterline) {
      parentOp->emitError()
          << "cannot initialize ILP variable for secret.generic argument "
          << arg.getArgNumber() << " from mgmt.mgmt level " << initialLevel
          << "; expected level in [0, " << state.bootstrapWaterline << "]";
      return failure();
    }
    if (initialScale < state.sw || initialScale > state.scaleMax) {
      parentOp->emitError()
          << "cannot initialize ILP variable for secret.generic argument "
          << arg.getArgNumber() << " from mgmt.mgmt scale " << initialScale
          << "; expected scale in [" << state.sw << ", " << state.scaleMax
          << "]";
      return failure();
    }

    std::stringstream ssLevel;
    ssLevel << "levelArg" << arg.getArgNumber();
    auto levelVar =
        state.addValueLevelVar(0, state.bootstrapWaterline, ssLevel.str());
    state.valueLevelVars.insert(std::make_pair(arg, levelVar));

    std::stringstream ssScale;
    ssScale << "scaleArg" << arg.getArgNumber();
    auto scaleVar = state.addScaleVar(ssScale.str());
    state.valueScaleVars.insert(std::make_pair(arg, scaleVar));

    state.model.AddLinearConstraint(
        levelVar == initialLevel,
        "initLevelArg" + std::to_string(arg.getArgNumber()));
    state.model.AddLinearConstraint(
        scaleVar == initialScale,
        "initScaleArg" + std::to_string(arg.getArgNumber()));
  }
  return success();
}

static bool hasSecretResult(Operation& op, DataFlowSolver* solver) {
  return llvm::any_of(op.getResults(), [&](OpResult result) {
    return isSecret(result, solver);
  });
}

static bool shouldTrackOperation(Operation& op, DataFlowSolver* solver) {
  return !isa<secret::YieldOp>(op) && hasSecretResult(op, solver);
}

static void addOperationDecisionVariables(ILPModelState& state, Operation* op,
                                          const std::string& opName) {
  auto inputLevelVar =
      state.addLevelVar(0, state.bootstrapWaterline, "inputLevel" + opName);
  state.inputLevelVars.insert(std::make_pair(op, inputLevelVar));
  auto inputScaleVar = state.addScaleVar("inputScale" + opName);
  state.inputScaleVars.insert(std::make_pair(op, inputScaleVar));
  auto nodeRescaleVar =
      state.addLevelVar(0, state.bootstrapWaterline, "nodeRescale" + opName);
  state.nodeRescaleVars.insert(std::make_pair(op, nodeRescaleVar));
  auto bootstrapVar = state.model.AddBinaryVariable("bootstrap" + opName);
  state.bootstrapVars.insert(std::make_pair(op, bootstrapVar));

  if (state.levelOnly) {
    state.model.AddLinearConstraint(inputScaleVar == state.sw,
                                    "levelOnlyInputScale" + opName);
  }
}

static void addOperationResultVariables(ILPModelState& state, Operation* op,
                                        const std::string& opName) {
  for (OpResult result : op->getResults()) {
    if (!isSecret(result, state.solver)) continue;

    std::stringstream ssLevel;
    ssLevel << "level" << opName << result.getResultNumber();
    auto levelVar =
        state.addValueLevelVar(0, state.bootstrapWaterline, ssLevel.str());
    state.valueLevelVars.insert(std::make_pair(result, levelVar));

    std::stringstream ssScale;
    ssScale << "scale" << opName << result.getResultNumber();
    auto scaleVar = state.addScaleVar(ssScale.str());
    state.valueScaleVars.insert(std::make_pair(result, scaleVar));
    if (state.levelOnly) {
      state.model.AddLinearConstraint(
          scaleVar == state.sw, "levelOnlyResultScale" + opName +
                                    std::to_string(result.getResultNumber()));
    }
  }
}

static void addTrackedOperationVariables(ILPModelState& state) {
  for (Operation& op : state.body->getOperations()) {
    if (!shouldTrackOperation(op, state.solver)) continue;

    state.trackedOps.push_back(&op);
    std::string opName = state.uniqueName(&op);
    addOperationDecisionVariables(state, &op, opName);
    addOperationResultVariables(state, &op, opName);
  }
}

// Add constraints for one producer value flowing into one operand use of an op.
// The edge rescale variable models management inserted before the consumer.
// Non-multiplication ops share one input scale across all operands; CKKS
// multiplications use per-edge scales so their combined scale can be modeled
// separately in addMultiplicationInputScaleConstraint.
static void addSingleOperandEdgeConstraints(ILPModelState& state, Operation* op,
                                            OpOperand& operandUse,
                                            const std::string& opName) {
  Value operand = operandUse.get();
  if (!isSecret(operand, state.solver)) return;
  if (!state.valueLevelVars.contains(operand)) return;

  std::stringstream ss;
  ss << opName << "Op" << operandUse.getOperandNumber();
  std::string edgeName = ss.str();

  auto inputLevelVar = state.inputLevelVars.at(op);
  auto inputScaleVar = state.inputScaleVars.at(op);
  auto edgeRescaleVar =
      state.addLevelVar(0, state.bootstrapWaterline, "edgeRescale" + edgeName);
  state.edgeRescaleVars.insert(std::make_pair(&operandUse, edgeRescaleVar));

  math_opt::Variable edgeScaleVar =
      (!state.levelOnly && isMultiplication(op))
          ? state.addScaleVar("edgeScale" + edgeName)
          : inputScaleVar;
  state.edgeScaleVars.insert(std::make_pair(&operandUse, edgeScaleVar));

  state.model.AddLinearConstraint(
      inputLevelVar <= state.valueLevelVars.at(operand) - edgeRescaleVar,
      "edgeLevel" + edgeName);
  state.model.AddLinearConstraint(
      edgeScaleVar >=
          state.valueScaleVars.at(operand) - state.sf * edgeRescaleVar,
      "edgeScale" + edgeName);
  if (state.levelOnly) {
    state.model.AddLinearConstraint(edgeScaleVar == state.sw,
                                    "levelOnlyEdgeScale" + edgeName);
  }
}

// Add the CKKS multiplication scale-composition constraint. After each operand
// edge chooses its aligned scale, the multiplication's raw input scale is the
// sum of those operand scales; plaintext constants contribute the waterline
// scale. Result/output scale constraints are added by
// addNodeTransitionConstraints.
static void addMultiplicationInputScaleConstraint(ILPModelState& state,
                                                  Operation* op,
                                                  const std::string& opName) {
  if (state.levelOnly || !isMultiplication(op)) return;

  math_opt::LinearExpression inputScale;
  for (OpOperand& operandUse : op->getOpOperands()) {
    Value operand = operandUse.get();
    if (isSecret(operand, state.solver) &&
        state.edgeScaleVars.contains(&operandUse)) {
      inputScale += state.edgeScaleVars.at(&operandUse);
    } else if (isConstantLike(operand)) {
      inputScale += state.sw;
    }
  }
  state.model.AddLinearConstraint(state.inputScaleVars.at(op) == inputScale,
                                  "mulInputScale" + opName);
}

static void addOperandEdgeConstraints(ILPModelState& state) {
  for (Operation* op : state.trackedOps) {
    std::string opName = state.uniqueName(op);
    for (OpOperand& operandUse : op->getOpOperands()) {
      addSingleOperandEdgeConstraints(state, op, operandUse, opName);
    }
    addMultiplicationInputScaleConstraint(state, op, opName);
  }
}

// Add output-boundary constraints for values yielded from secret.generic.
// When the corresponding generic result carries an explicit mgmt.mgmt attr,
// the yielded value's level is pinned to the annotated level, and (in CKKS
// mode) a nonzero annotated scale pins the yielded value's scale.
static LogicalResult addYieldConstraints(ILPModelState& state) {
  auto genericOp = cast<secret::GenericOp>(state.body->getParentOp());
  for (Operation& op : state.body->getOperations()) {
    auto yieldOp = dyn_cast<secret::YieldOp>(op);
    if (!yieldOp) continue;
    for (auto [index, operand] : llvm::enumerate(yieldOp->getOperands())) {
      if (!isSecret(operand, state.solver)) continue;
      if (!state.valueLevelVars.contains(operand)) continue;

      mgmt::MgmtAttr mgmtAttr =
          mgmt::findMgmtAttrAssociatedWith(genericOp.getResult(index));
      if (!mgmtAttr) continue;

      int resultLevel = mgmtAttr.getLevel();
      if (resultLevel < 0 || resultLevel > state.bootstrapWaterline) {
        genericOp->emitError()
            << "cannot constrain yielded value " << index
            << " from secret.generic result mgmt.mgmt level " << resultLevel
            << "; expected level in [0, " << state.bootstrapWaterline << "]";
        return failure();
      }
      state.model.AddLinearConstraint(
          state.valueLevelVars.at(operand) == resultLevel,
          "yieldResultLevel" + std::to_string(index));

      if (state.levelOnly || mgmtAttr.getScale() == 0) continue;
      if (!state.valueScaleVars.contains(operand)) continue;

      int resultScale = mgmtAttr.getScale();
      if (resultScale < state.sw || resultScale > state.scaleMax) {
        genericOp->emitError() << "cannot constrain yielded value " << index
                               << " from secret.generic result mgmt.mgmt scale "
                               << resultScale << "; expected scale in ["
                               << state.sw << ", " << state.scaleMax << "]";
        return failure();
      }
      state.model.AddLinearConstraint(
          state.valueScaleVars.at(operand) == resultScale,
          "yieldResultScale" + std::to_string(index));
    }
  }
  return success();
}

// Add constraints for each tracked op's result state after the op and any
// management chosen on the node. This is the result/output counterpart to the
// edge constraints: it relates the op's input level/scale to each secret result
// through either a direct transition or a bootstrap transition.
static void addNodeTransitionConstraints(ILPModelState& state,
                                         int bootstrapLevelLowerBound) {
  for (Operation* op : state.trackedOps) {
    std::string opName = state.uniqueName(op);
    auto inputLevelVar = state.inputLevelVars.at(op);
    auto inputScaleVar = state.inputScaleVars.at(op);
    auto nodeRescaleVar = state.nodeRescaleVars.at(op);
    auto bootstrapVar = state.bootstrapVars.at(op);
    int intrinsicLevelConsumption =
        state.levelOnly && isMultiplication(op) ? 1 : 0;

    for (OpResult result : op->getResults()) {
      if (!isSecret(result, state.solver)) continue;
      if (!state.valueLevelVars.contains(result)) continue;

      auto outputLevelVar = state.valueLevelVars.at(result);
      auto outputScaleVar = state.valueScaleVars.at(result);
      std::stringstream ss;
      ss << opName << "Result" << result.getResultNumber();

      // If bootstrap is false, this is the direct Orbit rescale/modswitch
      // transition. Level-only modswitch is allowed by the <= relation without
      // charging nodeRescaleVar.
      state.model.AddLinearConstraint(
          outputLevelVar <= inputLevelVar - intrinsicLevelConsumption -
                                nodeRescaleVar + state.bigM * bootstrapVar,
          "nodeDirectLevel" + ss.str());
      state.model.AddLinearConstraint(
          outputScaleVar >= inputScaleVar - state.sf * nodeRescaleVar -
                                state.scaleBigM * bootstrapVar,
          "nodeDirectScale" + ss.str());

      // If bootstrap is true, input must be bootstrappable and output must be
      // in Orbit's bootstrapped level/scale range.
      state.model.AddLinearConstraint(
          inputScaleVar <=
              state.sf * (inputLevelVar - bootstrapLevelLowerBound + 1) +
                  state.scaleBigM * (1 - bootstrapVar),
          "bootstrapInputFeasible" + ss.str());
      state.model.AddLinearConstraint(
          outputLevelVar >=
              (bootstrapLevelLowerBound + 1) - state.bigM * (1 - bootstrapVar),
          "bootstrapOutputLevel" + ss.str());
      state.model.AddLinearConstraint(
          outputScaleVar >= state.sf - state.scaleBigM * (1 - bootstrapVar),
          "bootstrapOutputScale" + ss.str());
    }
  }
}

static void addObjective(ILPModelState& state, const OpCostModel& costModel) {
  math_opt::LinearExpression objective;
  for (auto& [op, bootstrapVar] : state.bootstrapVars) {
    objective += costModel.bootstrapCost * bootstrapVar;
  }
  for (auto& [op, rescaleVar] : state.nodeRescaleVars) {
    objective += costModel.rescaleCost * rescaleVar;
  }
  for (auto& [operand, rescaleVar] : state.edgeRescaleVars) {
    objective += costModel.rescaleCost * rescaleVar;
  }
  // Level-dependent op latency: each tracked op is charged
  // slope * inputLevel + intercept for its cost class, so the solver prefers
  // to run expensive ops (muls, rotations) at low levels.
  if (costModel.hasLevelCosts) {
    for (Operation* op : state.trackedOps) {
      auto cost = levelCostForOp(op, state.solver, costModel);
      if (!cost.has_value()) continue;
      objective += cost->slope * state.inputLevelVars.at(op) + cost->intercept;
    }
  }
  // Tie-breaker on value (result) levels: level constraints are one-sided, so
  // among equal-cost solutions the solver could pick gratuitously low levels
  // (free modswitches decoded as spurious level_reduce ops). The small
  // negative weight prefers the highest feasible level for each value. Op
  // *input* levels are separate variables and get real downward pressure from
  // the level-dependent latency terms above, so the two do not conflict.
  for (auto& [value, levelVar] : state.valueLevelVars) {
    objective += -0.001 * levelVar;
  }
  state.model.Minimize(objective);
}

static void populateSolution(
    const math_opt::SolveResult& result, ILPModelState& state,
    llvm::DenseMap<Operation*, bool>& solution,
    llvm::DenseMap<Value, int>& solutionLevelBeforeBootstrap,
    llvm::DenseMap<Value, int>& solutionLevelAfterBootstrap,
    llvm::SmallVector<ILPBootstrapPlacementAnalysis::NodeManagement, 32>&
        nodeManagement,
    llvm::SmallVector<ILPBootstrapPlacementAnalysis::EdgeManagement, 32>&
        edgeManagement) {
  auto varMap = result.variable_values();
  for (Operation* op : state.trackedOps) {
    bool useBootstrap = varMap.at(state.bootstrapVars.at(op)) > 0.5;
    solution.insert(std::make_pair(op, useBootstrap));

    int inputLevel = roundedValue(varMap, state.inputLevelVars.at(op));
    int inputScale = roundedValue(varMap, state.inputScaleVars.at(op));
    for (OpResult result : op->getResults()) {
      if (!isSecret(result, state.solver)) continue;
      int outputLevel = roundedValue(varMap, state.valueLevelVars.at(result));
      int outputScale = roundedValue(varMap, state.valueScaleVars.at(result));
      solutionLevelBeforeBootstrap.insert(std::make_pair(result, inputLevel));
      solutionLevelAfterBootstrap.insert(std::make_pair(result, outputLevel));
      nodeManagement.push_back({result, inputLevel, inputScale, outputLevel,
                                outputScale, useBootstrap});
    }
  }

  for (Operation* op : state.trackedOps) {
    int targetLevel = roundedValue(varMap, state.inputLevelVars.at(op));
    for (OpOperand& operandUse : op->getOpOperands()) {
      if (!state.edgeScaleVars.contains(&operandUse)) continue;
      Value operand = operandUse.get();
      int sourceLevel = roundedValue(varMap, state.valueLevelVars.at(operand));
      int sourceScale = roundedValue(varMap, state.valueScaleVars.at(operand));
      int targetScale =
          roundedValue(varMap, state.edgeScaleVars.at(&operandUse));
      edgeManagement.push_back({op, operandUse.getOperandNumber(), sourceLevel,
                                sourceScale, targetLevel, targetScale});
    }
  }
}

LogicalResult ILPBootstrapPlacementAnalysis::solve() {
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) return failure();

  ILPModelState state(genericOp.getBody(), solver, bootstrapWaterline,
                      scaleWaterline, scaleFactorBits, scaleMode);

  if (failed(addBodyArgumentVariables(state))) return failure();
  addTrackedOperationVariables(state);
  addOperandEdgeConstraints(state);
  if (failed(addYieldConstraints(state))) return failure();
  addNodeTransitionConstraints(state, bootstrapLevelLowerBound);
  addObjective(state, costModel);

  LLVM_DEBUG({
    std::stringstream ss;
    ss << state.model;
    llvm::dbgs() << "--- ILP model ---\n" << ss.str() << "--- end model ---\n";
  });

  // Solve to a 1% relative optimality gap, matching Orbit's solver
  // configuration (Gurobi MIPGap / CBC gapRel = 0.01). Proving full optimality
  // often dominates solve time on large instances while improving the
  // objective by less than measurement noise in the profiled cost models. On
  // small instances the solver typically closes the gap entirely, so this
  // rarely changes the chosen placement.
  constexpr double kRelativeMipGap = 0.01;
  math_opt::SolveArguments solveArgs;
  solveArgs.parameters.relative_gap_tolerance = kRelativeMipGap;

  const absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(state.model, math_opt::SolverType::kGscip, solveArgs);
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
      llvm::errs() << "No feasible solution found (the problem may be "
                      "infeasible). Termination status code: "
                   << static_cast<int>(result.termination.reason) << "\n";
      return failure();
  }

  populateSolution(result, state, solution, solutionLevelBeforeBootstrap,
                   solutionLevelAfterBootstrap, nodeManagement, edgeManagement);

  return success();
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
  os << "bootstrap waterline: " << bootstrapWaterline << "\n";
  os << "scale waterline: " << scaleWaterline << "\n";
  os << "scale factor bits: " << scaleFactorBits << "\n\n";

  for (BlockArgument arg : body->getArguments()) {
    auto it = solutionLevelAfterBootstrap.find(arg);
    if (it != solutionLevelAfterBootstrap.end()) {
      os << "  ";
      arg.print(os);
      os << "  level(after): " << it->second << "\n";
    }
  }

  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(op)) continue;
    bool insertBootstrap = solution.lookup(&op);
    os << "  ";
    op.print(os);
    os << "  bootstrap=" << (insertBootstrap ? "yes" : "no");
    for (const auto& placement : nodeManagement) {
      if (placement.value.getDefiningOp() == &op) {
        os << " transition=(" << placement.inputLevel << ","
           << placement.inputScale << ")->(" << placement.outputLevel << ","
           << placement.outputScale << ")";
      }
    }
    os << "\n";
  }
  os << "--- end solution ---\n";
}

}  // namespace heir
}  // namespace mlir
