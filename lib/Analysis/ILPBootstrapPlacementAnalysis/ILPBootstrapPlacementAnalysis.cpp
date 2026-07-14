#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "lib/Analysis/ILPBootstrapPlacementAnalysis/OpGrouping.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"                // from @llvm-project
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
// ILP formulation (per group of ops; see OpGrouping.h — grouped ops share one
// set of variables, and every objective term is scaled by the group's
// multiplicity):
// - Variables:
//   * level[value], scale[value] (in bits) for each secret SSA value class
//   * input_level[group], input_scale[group] for each tracked group
//   * bootstrap[group], node_rescale[group] for management after a group
//   * edge_rescale[edge] and, for CKKS multiplication operands,
//     edge_scale[edge] for management before a consuming group, where an edge
//     merges all operand uses of one producer class by one group
// - Initialization:
//   * secret.generic body args are initialized from associated mgmt.mgmt attrs
//     when present, otherwise from (bootstrapWaterline, Sw)
// - Constraints:
//   * bounds: levels are 0..bootstrapWaterline; CKKS scales are Sw..scaleMax
//   * level-only mode fixes all live scales to Sw
//   * operand edges allow level/scale reduction before the consuming group
//   * non-multiplication operands share the consumer's input_scale
//   * CKKS multiplication input_scale is the sum of operand edge scales;
//     plaintext constants contribute Sw
//   * node transitions relate each group's input state to its result state by
//     either direct rescale/modswitch management or a bootstrap transition
//   * a yielded value is pinned to the level annotated on its secret.generic
//     result by an mgmt.mgmt attr, and to the annotated scale when nonzero
// - Objective: minimize total bootstrap and rescale cost. With a per-level cost
//   model, also charge each tracked op its latency at its input level. A tiny
//   per-level term breaks ties toward higher levels for output values.
//
// Large circuits are additionally cut at single-input single-output (SISO)
// boundaries (Orbit's partitioning): each partition is solved independently
// under enumerated boundary (level, scale) states, producing a transfer table
// of boundary-in -> boundary-out costs, and a dynamic program stitches the
// per-partition solutions into a global placement.

namespace math_opt = ::operations_research::math_opt;

#define DEBUG_TYPE "ilp-bootstrap-placement"

namespace mlir {
namespace heir {

using ScaleMode = ILPBootstrapPlacementAnalysis::ScaleMode;
using Options = ILPBootstrapPlacementAnalysis::Options;
using NodeManagement = ILPBootstrapPlacementAnalysis::NodeManagement;
using EdgeManagement = ILPBootstrapPlacementAnalysis::EdgeManagement;

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

// The smallest strictly positive cost that changes with a value's level: one
// rescale, or one level's worth of any op-latency slope. The value-level
// tie-breaker's total magnitude is capped below this so it can only break
// ties, never outweigh a real level-dependent cost. Intercepts and the
// bootstrap constant are excluded because they do not vary with a value
// level at the margin.
static double minLevelMarginalCost(const OpCostModel& costModel) {
  double minCost = std::numeric_limits<double>::infinity();
  auto consider = [&](double value) {
    if (value > 0 && value < minCost) minCost = value;
  };
  consider(costModel.rescaleCost);
  if (costModel.hasLevelCosts) {
    for (const LinearCost* cost :
         {&costModel.addCtCt, &costModel.addCtPt, &costModel.mulCtCt,
          &costModel.mulCtPt, &costModel.rotate, &costModel.negate}) {
      consider(std::abs(cost->slope));
    }
  }
  return std::isinf(minCost) ? 1.0 : minCost;
}

namespace {

struct ValueState {
  int level = 0;
  int scale = 0;
};

// Boundary specification for one partition solve.
struct PartitionBoundary {
  SmallVector<std::pair<Value, ValueState>, 4> pinnedInputs;
  Value outputValue;
  int outputLevel = 0;
  // Yield pins apply only to the partition containing the yield.
  bool applyYieldConstraints = true;
  // Number of original ops in the partition, used to weigh the boundary
  // output-scale pressure term.
  int sizeInOps = 0;
};

// One merged operand edge: all uses of one producer value class by one
// consumer group share these decision variables.
struct GroupEdge {
  math_opt::Variable rescaleVar;
  math_opt::Variable scaleVar;
  int weight = 0;
};

struct ILPModelState {
  ILPModelState(Block* body, DataFlowSolver* solver, const Options& options,
                const OpGrouping& grouping, int groupBegin, int groupEnd,
                const PartitionBoundary& boundary)
      : model("ILPBootstrapPlacementAnalysis"),
        body(body),
        solver(solver),
        bootstrapWaterline(options.bootstrapWaterline),
        levelOnly(options.scaleMode == ScaleMode::kLevelOnly),
        sw(options.scaleWaterline),
        sf(options.scaleFactorBits),
        scaleMax(levelOnly ? sw : sf + 2 * sw),
        bigM(bootstrapWaterline + 1),
        scaleBigM(4 * scaleMax + sf * (bootstrapWaterline + 1)),
        grouping(grouping),
        groupBegin(groupBegin),
        groupEnd(groupEnd),
        boundary(boundary) {}

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

  Value canon(Value value) const { return grouping.canonicalValue(value); }

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
  const OpGrouping& grouping;
  int groupBegin;
  int groupEnd;
  const PartitionBoundary& boundary;
  int nextOpaqueId = 0;
  llvm::DenseMap<Operation*, int> opaqueIds;
  // Value variables are keyed by canonical values (see OpGrouping::valueRep).
  llvm::DenseMap<Value, math_opt::Variable> valueLevelVars;
  llvm::DenseMap<Value, math_opt::Variable> valueScaleVars;
  // Input variables are aliased to every member op of a group.
  llvm::DenseMap<Operation*, math_opt::Variable> inputLevelVars;
  llvm::DenseMap<Operation*, math_opt::Variable> inputScaleVars;
  // Management variables are keyed by the group representative.
  llvm::DenseMap<Operation*, math_opt::Variable> nodeRescaleVars;
  llvm::DenseMap<Operation*, math_opt::Variable> bootstrapVars;
  // Edge variables are aliased to every operand use merged into the edge.
  llvm::DenseMap<OpOperand*, math_opt::Variable> edgeScaleVars;
  llvm::DenseMap<OpOperand*, math_opt::Variable> edgeRescaleVars;
  SmallVector<GroupEdge> edges;
  llvm::DenseMap<std::pair<Value, int>, int> edgeIndex;
  SmallVector<Operation*> trackedOps;
};

// One entry of the partition dynamic-programming transfer table: the decoded
// mgmt decisions and cost for one partition solved under one (input-state,
// output-level) boundary. The DP selects one of these per partition and the
// analysis adopts the chosen chain.
struct PartitionSolution {
  double cost = 0;
  ValueState outState;
  std::pair<int, int> inKey;
  SmallVector<NodeManagement, 32> nodeManagement;
  SmallVector<EdgeManagement, 32> edgeManagement;
  llvm::DenseMap<Operation*, bool> bootstrapDecisions;
  llvm::DenseMap<Value, int> levelBefore;
  llvm::DenseMap<Value, int> levelAfter;
};

struct Partition {
  int groupBegin;
  int groupEnd;
  // Canonical value live across the cut after this partition; null on the
  // last partition.
  Value cutValue;
  int sizeInOps = 0;
};

}  // namespace

// Determine the initial (level, scale) of each secret generic argument from
// associated mgmt.mgmt attrs, or the (bootstrapWaterline, Sw) defaults.
static LogicalResult computeArgInitStates(
    Block* body, DataFlowSolver* solver, const Options& options,
    SmallVector<std::pair<Value, ValueState>, 4>& pinnedInputs) {
  bool levelOnly = options.scaleMode == ScaleMode::kLevelOnly;
  int sw = options.scaleWaterline;
  int scaleMax = levelOnly ? sw : options.scaleFactorBits + 2 * sw;
  for (BlockArgument arg : body->getArguments()) {
    if (!isSecret(arg, solver)) continue;

    int initialLevel = options.bootstrapWaterline;
    int initialScale = sw;
    if (mgmt::MgmtAttr mgmtAttr = mgmt::findMgmtAttrAssociatedWith(arg)) {
      initialLevel = mgmtAttr.getLevel();
      if (!levelOnly && mgmtAttr.getScale() != 0) {
        initialScale = mgmtAttr.getScale();
      }
    }

    Operation* parentOp = body->getParentOp();
    if (initialLevel < 0 || initialLevel > options.bootstrapWaterline) {
      parentOp->emitError()
          << "cannot initialize ILP variable for secret.generic argument "
          << arg.getArgNumber() << " from mgmt.mgmt level " << initialLevel
          << "; expected level in [0, " << options.bootstrapWaterline << "]";
      return failure();
    }
    if (initialScale < sw || initialScale > scaleMax) {
      parentOp->emitError()
          << "cannot initialize ILP variable for secret.generic argument "
          << arg.getArgNumber() << " from mgmt.mgmt scale " << initialScale
          << "; expected scale in [" << sw << ", " << scaleMax << "]";
      return failure();
    }
    pinnedInputs.push_back({arg, {initialLevel, initialScale}});
  }
  return success();
}

static void addPinnedInputVariables(ILPModelState& state) {
  int index = 0;
  for (auto& [value, valueState] : state.boundary.pinnedInputs) {
    std::string name = "Pinned" + std::to_string(index++);
    auto levelVar =
        state.addValueLevelVar(0, state.bootstrapWaterline, "level" + name);
    state.valueLevelVars.insert(std::make_pair(value, levelVar));
    auto scaleVar = state.addScaleVar("scale" + name);
    state.valueScaleVars.insert(std::make_pair(value, scaleVar));

    state.model.AddLinearConstraint(levelVar == valueState.level,
                                    "initLevel" + name);
    state.model.AddLinearConstraint(scaleVar == valueState.scale,
                                    "initScale" + name);
  }
}

static void addTrackedGroupVariables(ILPModelState& state) {
  for (int gi = state.groupBegin; gi < state.groupEnd; ++gi) {
    const OpGroup& group = state.grouping.groups[gi];
    Operation* rep = group.representative;
    std::string opName = state.uniqueName(rep);

    auto inputLevelVar =
        state.addLevelVar(0, state.bootstrapWaterline, "inputLevel" + opName);
    auto inputScaleVar = state.addScaleVar("inputScale" + opName);
    auto nodeRescaleVar =
        state.addLevelVar(0, state.bootstrapWaterline, "nodeRescale" + opName);
    auto bootstrapVar = state.model.AddBinaryVariable("bootstrap" + opName);
    state.nodeRescaleVars.insert(std::make_pair(rep, nodeRescaleVar));
    state.bootstrapVars.insert(std::make_pair(rep, bootstrapVar));
    for (Operation* member : group.members) {
      state.inputLevelVars.insert(std::make_pair(member, inputLevelVar));
      state.inputScaleVars.insert(std::make_pair(member, inputScaleVar));
      state.trackedOps.push_back(member);
    }

    if (state.levelOnly) {
      state.model.AddLinearConstraint(inputScaleVar == state.sw,
                                      "levelOnlyInputScale" + opName);
    }

    for (OpResult result : rep->getResults()) {
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
}

// Add constraints for the operand edges of each group. All uses of one
// producer value class by one group share a single edge: the edge rescale
// variable models management inserted before the consumers, charged once per
// original use in the objective. Non-multiplication groups share one input
// scale across all operands; CKKS multiplications use per-edge scales so
// their combined scale can be modeled separately below.
static void addGroupEdgeConstraints(ILPModelState& state) {
  for (int gi = state.groupBegin; gi < state.groupEnd; ++gi) {
    const OpGroup& group = state.grouping.groups[gi];
    Operation* rep = group.representative;
    std::string opName = state.uniqueName(rep);
    auto inputLevelVar = state.inputLevelVars.at(rep);
    auto inputScaleVar = state.inputScaleVars.at(rep);

    for (Operation* member : group.members) {
      // Weight counts distinct (producer class, consumer op) connections, so
      // repeated uses of one producer by one op (e.g. squaring x*x) are one
      // edge, matching Orbit's DiGraph edge model. Every operand use still
      // registers the shared edge variables so the CKKS scale-composition sum
      // and the decoder can see each slot.
      DenseSet<Value> countedForMember;
      for (OpOperand& operandUse : member->getOpOperands()) {
        Value operand = operandUse.get();
        if (!isSecret(operand, state.solver)) continue;
        if (Operation* def = operand.getDefiningOp()) {
          auto it = state.grouping.groupIdOf.find(def);
          if (it != state.grouping.groupIdOf.end() && it->second == gi)
            continue;  // interior edge of an addition tree
        }
        Value key = state.canon(operand);
        if (!state.valueLevelVars.contains(key)) continue;

        auto [entry, inserted] = state.edgeIndex.try_emplace(
            std::make_pair(key, gi), state.edges.size());
        if (inserted) {
          std::string edgeName =
              opName + "Edge" + std::to_string(state.edges.size());
          auto edgeRescaleVar = state.addLevelVar(0, state.bootstrapWaterline,
                                                  "edgeRescale" + edgeName);
          math_opt::Variable edgeScaleVar =
              (!state.levelOnly && group.isMultiplication)
                  ? state.addScaleVar("edgeScale" + edgeName)
                  : inputScaleVar;

          state.model.AddLinearConstraint(
              inputLevelVar <= state.valueLevelVars.at(key) - edgeRescaleVar,
              "edgeLevel" + edgeName);
          state.model.AddLinearConstraint(
              edgeScaleVar >=
                  state.valueScaleVars.at(key) - state.sf * edgeRescaleVar,
              "edgeScale" + edgeName);
          if (state.levelOnly) {
            state.model.AddLinearConstraint(edgeScaleVar == state.sw,
                                            "levelOnlyEdgeScale" + edgeName);
          }
          state.edges.push_back({edgeRescaleVar, edgeScaleVar, 0});
        }
        GroupEdge& edge = state.edges[entry->second];
        if (countedForMember.insert(key).second) edge.weight += 1;
        state.edgeRescaleVars.insert(
            std::make_pair(&operandUse, edge.rescaleVar));
        state.edgeScaleVars.insert(std::make_pair(&operandUse, edge.scaleVar));
      }
    }

    // The CKKS multiplication scale-composition constraint: the group's raw
    // input scale is the sum of the representative's operand edge scales;
    // plaintext constants contribute the waterline scale. Any merged member
    // satisfies the same constraint by symmetry.
    if (!state.levelOnly && group.isMultiplication) {
      math_opt::LinearExpression inputScale;
      for (OpOperand& operandUse : rep->getOpOperands()) {
        Value operand = operandUse.get();
        if (isSecret(operand, state.solver) &&
            state.edgeScaleVars.contains(&operandUse)) {
          inputScale += state.edgeScaleVars.at(&operandUse);
        } else if (isConstantLike(operand)) {
          inputScale += state.sw;
        }
      }
      state.model.AddLinearConstraint(inputScaleVar == inputScale,
                                      "mulInputScale" + opName);
    }
  }
}

// Add output-boundary constraints for values yielded from secret.generic.
// When the corresponding generic result carries an explicit mgmt.mgmt attr,
// the yielded value's level is pinned to the annotated level, and (in CKKS
// mode) a nonzero annotated scale pins the yielded value's scale.
static LogicalResult addYieldConstraints(ILPModelState& state) {
  if (!state.boundary.applyYieldConstraints) return success();
  auto genericOp = cast<secret::GenericOp>(state.body->getParentOp());
  for (Operation& op : state.body->getOperations()) {
    auto yieldOp = dyn_cast<secret::YieldOp>(op);
    if (!yieldOp) continue;
    for (auto [index, operand] : llvm::enumerate(yieldOp->getOperands())) {
      if (!isSecret(operand, state.solver)) continue;
      Value key = state.canon(operand);
      if (!state.valueLevelVars.contains(key)) continue;

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
          state.valueLevelVars.at(key) == resultLevel,
          "yieldResultLevel" + std::to_string(index));

      if (state.levelOnly || mgmtAttr.getScale() == 0) continue;
      if (!state.valueScaleVars.contains(key)) continue;

      int resultScale = mgmtAttr.getScale();
      if (resultScale < state.sw || resultScale > state.scaleMax) {
        genericOp->emitError() << "cannot constrain yielded value " << index
                               << " from secret.generic result mgmt.mgmt scale "
                               << resultScale << "; expected scale in ["
                               << state.sw << ", " << state.scaleMax << "]";
        return failure();
      }
      state.model.AddLinearConstraint(
          state.valueScaleVars.at(key) == resultScale,
          "yieldResultScale" + std::to_string(index));
    }
  }
  return success();
}

// Pin the partition's boundary output level during boundary enumeration.
static void addOutputPinConstraints(ILPModelState& state) {
  if (!state.boundary.outputValue) return;
  Value key = state.canon(state.boundary.outputValue);
  state.model.AddLinearConstraint(
      state.valueLevelVars.at(key) == state.boundary.outputLevel,
      "boundaryOutLevel");
  if (!state.levelOnly) {
    // Backend feasibility at a partition boundary, matching Orbit's output
    // scale/level relation for Lattigo: scale <= Sf * (level + 1) - margin.
    constexpr int kBoundaryScaleMargin = 7;
    state.model.AddLinearConstraint(
        state.valueScaleVars.at(key) <=
            state.sf * (state.boundary.outputLevel + 1) - kBoundaryScaleMargin,
        "boundaryOutScale");
  }
}

// Add constraints for each group's result state after the group and any
// management chosen on the node. This is the result/output counterpart to the
// edge constraints: it relates the group's input level/scale to its result
// through either a direct transition or a bootstrap transition.
static void addNodeTransitionConstraints(ILPModelState& state,
                                         int bootstrapLevelLowerBound) {
  for (int gi = state.groupBegin; gi < state.groupEnd; ++gi) {
    const OpGroup& group = state.grouping.groups[gi];
    Operation* rep = group.representative;
    std::string opName = state.uniqueName(rep);
    auto inputLevelVar = state.inputLevelVars.at(rep);
    auto inputScaleVar = state.inputScaleVars.at(rep);
    auto nodeRescaleVar = state.nodeRescaleVars.at(rep);
    auto bootstrapVar = state.bootstrapVars.at(rep);
    int intrinsicLevelConsumption =
        state.levelOnly && group.isMultiplication ? 1 : 0;

    for (OpResult result : rep->getResults()) {
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
  // Management decisions are charged once per merged management site.
  for (int gi = state.groupBegin; gi < state.groupEnd; ++gi) {
    const OpGroup& group = state.grouping.groups[gi];
    Operation* rep = group.representative;
    objective +=
        group.weight * costModel.bootstrapCost * state.bootstrapVars.at(rep);
    objective +=
        group.weight * costModel.rescaleCost * state.nodeRescaleVars.at(rep);
  }
  for (const GroupEdge& edge : state.edges) {
    objective += edge.weight * costModel.rescaleCost * edge.rescaleVar;
  }
  // Level-dependent op latency: each tracked op is charged
  // slope * inputLevel + intercept for its cost class, so the solver prefers
  // to run expensive ops (muls, rotations) at low levels. Members of a group
  // share one input level variable, so this sums member costs at that level.
  if (costModel.hasLevelCosts) {
    for (Operation* op : state.trackedOps) {
      auto cost = levelCostForOp(op, state.solver, costModel);
      if (!cost.has_value()) continue;
      objective += cost->slope * state.inputLevelVars.at(op) + cost->intercept;
    }
  }
  // Tie-breaker on value (result) levels: level constraints are one-sided, so
  // among equal-cost solutions the solver could pick gratuitously low levels
  // (free modswitches decoded as spurious level_reduce ops). A small negative
  // weight prefers the highest feasible level for each value. Op *input*
  // levels are separate variables and get real downward pressure from the
  // level-dependent latency terms above, so the two do not conflict.
  //
  // The per-value weight is nominally kEpsilon, but its total magnitude
  // (summed over every value level, each at most bootstrapWaterline) is capped
  // strictly below the smallest real level-marginal cost, so on large regions
  // it can only break ties and never outweigh a genuine cost difference. Small
  // regions keep the exact kEpsilon weight.
  constexpr double kEpsilon = 0.001;
  double numValues = state.valueLevelVars.size();
  double levelSpan = std::max(1, state.bootstrapWaterline);
  double tieBreakCeiling = 0.5 * minLevelMarginalCost(costModel);
  double perValueWeight = kEpsilon;
  if (kEpsilon * numValues * levelSpan > tieBreakCeiling) {
    perValueWeight = tieBreakCeiling / (numValues * levelSpan);
  }
  for (auto& [value, levelVar] : state.valueLevelVars) {
    objective += -perValueWeight * levelVar;
  }
  // At a partition boundary, prefer low output scale as a proxy for the
  // rescale work pushed onto downstream partitions (Orbit's 0.2-weighted
  // boundary term). Excluded from the recorded partition cost.
  if (state.boundary.outputValue && !state.levelOnly) {
    objective +=
        0.2 * costModel.rescaleCost * state.boundary.sizeInOps *
        state.valueScaleVars.at(state.canon(state.boundary.outputValue));
  }
  state.model.Minimize(objective);
}

// Extract the mgmt decisions for every member op of every group in range, and
// re-evaluate the exact objective share of this partition (excluding
// tie-break and boundary-pressure terms).
static void populatePartitionSolution(const math_opt::SolveResult& result,
                                      ILPModelState& state,
                                      const OpCostModel& costModel,
                                      PartitionSolution& soln) {
  auto varMap = result.variable_values();
  double cost = 0;
  for (int gi = state.groupBegin; gi < state.groupEnd; ++gi) {
    const OpGroup& group = state.grouping.groups[gi];
    Operation* rep = group.representative;
    bool useBootstrap = varMap.at(state.bootstrapVars.at(rep)) > 0.5;
    int nodeRescales = roundedValue(varMap, state.nodeRescaleVars.at(rep));
    cost += group.weight * costModel.bootstrapCost * (useBootstrap ? 1 : 0);
    cost += group.weight * costModel.rescaleCost * nodeRescales;

    int inputLevel = roundedValue(varMap, state.inputLevelVars.at(rep));
    int inputScale = roundedValue(varMap, state.inputScaleVars.at(rep));
    llvm::DenseSet<Value> interior(group.interiorValues.begin(),
                                   group.interiorValues.end());
    for (Operation* member : group.members) {
      bool isManagementSite = false;
      for (OpResult result : member->getResults()) {
        if (!isSecret(result, state.solver)) continue;
        if (interior.contains(result)) {
          // Addition-tree interior values stay at the group's input state.
          soln.nodeManagement.push_back(
              {result, inputLevel, inputScale, inputLevel, inputScale, false});
          soln.levelBefore.insert(std::make_pair(result, inputLevel));
          soln.levelAfter.insert(std::make_pair(result, inputLevel));
          continue;
        }
        isManagementSite = true;
        Value key = state.canon(result);
        int outputLevel = roundedValue(varMap, state.valueLevelVars.at(key));
        int outputScale = roundedValue(varMap, state.valueScaleVars.at(key));
        soln.nodeManagement.push_back({result, inputLevel, inputScale,
                                       outputLevel, outputScale, useBootstrap});
        soln.levelBefore.insert(std::make_pair(result, inputLevel));
        soln.levelAfter.insert(std::make_pair(result, outputLevel));
      }
      soln.bootstrapDecisions.insert(
          std::make_pair(member, isManagementSite && useBootstrap));
    }
  }
  for (const GroupEdge& edge : state.edges) {
    cost += edge.weight * costModel.rescaleCost *
            roundedValue(varMap, edge.rescaleVar);
  }
  if (costModel.hasLevelCosts) {
    for (Operation* op : state.trackedOps) {
      auto opCost = levelCostForOp(op, state.solver, costModel);
      if (!opCost.has_value()) continue;
      cost +=
          opCost->slope * roundedValue(varMap, state.inputLevelVars.at(op)) +
          opCost->intercept;
    }
  }
  soln.cost = cost;

  for (Operation* op : state.trackedOps) {
    int targetLevel = roundedValue(varMap, state.inputLevelVars.at(op));
    // One transition per (op, producer class): repeated operand slots consume
    // the same managed value, so the decoder rescales the producer once and
    // points every matching slot at it.
    DenseSet<Value> emittedForOp;
    for (OpOperand& operandUse : op->getOpOperands()) {
      if (!state.edgeScaleVars.contains(&operandUse)) continue;
      Value key = state.canon(operandUse.get());
      if (!emittedForOp.insert(key).second) continue;
      int sourceLevel = roundedValue(varMap, state.valueLevelVars.at(key));
      int sourceScale = roundedValue(varMap, state.valueScaleVars.at(key));
      int targetScale =
          roundedValue(varMap, state.edgeScaleVars.at(&operandUse));
      soln.edgeManagement.push_back({op, operandUse.getOperandNumber(),
                                     sourceLevel, sourceScale, targetLevel,
                                     targetScale});
    }
  }

  if (state.boundary.outputValue) {
    Value key = state.canon(state.boundary.outputValue);
    soln.outState.level = roundedValue(varMap, state.valueLevelVars.at(key));
    soln.outState.scale = roundedValue(varMap, state.valueScaleVars.at(key));
  }
}

// Build and solve the model for one partition under one boundary state.
// Returns failure only on hard errors; an infeasible boundary state leaves
// solutionOut empty.
static LogicalResult solvePartition(
    Block* body, DataFlowSolver* solver, const Options& options,
    const OpGrouping& grouping, int groupBegin, int groupEnd,
    const PartitionBoundary& boundary,
    std::optional<PartitionSolution>& solutionOut) {
  solutionOut.reset();
  ILPModelState state(body, solver, options, grouping, groupBegin, groupEnd,
                      boundary);

  addPinnedInputVariables(state);
  addTrackedGroupVariables(state);
  addGroupEdgeConstraints(state);
  if (failed(addYieldConstraints(state))) return failure();
  addNodeTransitionConstraints(state, options.bootstrapLevelLowerBound);
  addOutputPinConstraints(state);
  addObjective(state, options.costModel);

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
      // Infeasible under this boundary state; the caller decides whether that
      // is an error.
      LLVM_DEBUG(llvm::dbgs()
                 << "partition solve infeasible, termination status code: "
                 << static_cast<int>(result.termination.reason) << "\n");
      return success();
  }

  PartitionSolution soln;
  populatePartitionSolution(result, state, options.costModel, soln);
  solutionOut = std::move(soln);
  return success();
}

// Cut the group sequence at single-input single-output boundaries: positions
// where exactly one value class is live across the cut (Orbit's SISO
// partitioning).
static SmallVector<Partition> computePartitions(const OpGrouping& grouping,
                                                Block* body,
                                                DataFlowSolver* solver,
                                                int partitionMinSize) {
  int numGroups = grouping.groups.size();
  SmallVector<Partition> partitions;
  if (numGroups == 0) {
    partitions.push_back({0, 0, Value(), 0});
    return partitions;
  }

  // Producer position and last consumer position per canonical value class.
  llvm::DenseMap<Value, int> producerPos;
  llvm::DenseMap<Value, int> lastUsePos;
  for (BlockArgument arg : body->getArguments()) {
    if (isSecret(arg, solver)) producerPos[arg] = -1;
  }
  for (int gi = 0; gi < numGroups; ++gi) {
    for (Value result : grouping.groups[gi].resultValues) {
      producerPos[grouping.canonicalValue(result)] = gi;
    }
  }
  for (int gi = 0; gi < numGroups; ++gi) {
    for (Operation* member : grouping.groups[gi].members) {
      for (Value operand : member->getOperands()) {
        if (!isSecret(operand, solver)) continue;
        if (Operation* def = operand.getDefiningOp()) {
          auto it = grouping.groupIdOf.find(def);
          if (it != grouping.groupIdOf.end() && it->second == gi) continue;
        }
        Value key = grouping.canonicalValue(operand);
        if (!producerPos.contains(key)) continue;
        int& last = lastUsePos[key];
        last = std::max(last, gi);
      }
    }
  }
  for (Operation& op : body->getOperations()) {
    if (!isa<secret::YieldOp>(op)) continue;
    for (Value operand : op.getOperands()) {
      Value key = grouping.canonicalValue(operand);
      if (producerPos.contains(key)) lastUsePos[key] = INT_MAX;
    }
  }

  // Live-class count at each candidate cut (between positions p and p + 1).
  std::vector<int> delta(numGroups + 1, 0);
  for (auto& [key, prod] : producerPos) {
    auto it = lastUsePos.find(key);
    if (it == lastUsePos.end() || it->second <= prod) continue;
    int lo = std::max(prod, 0);
    int hi = std::min(it->second == INT_MAX ? numGroups - 1 : it->second - 1,
                      numGroups - 2);
    if (lo > hi) continue;
    delta[lo] += 1;
    delta[hi + 1] -= 1;
  }
  auto cutValueAt = [&](int p) -> Value {
    for (auto& [key, prod] : producerPos) {
      auto it = lastUsePos.find(key);
      if (it == lastUsePos.end()) continue;
      if (prod <= p && it->second > p) return key;
    }
    return Value();
  };

  SmallVector<int> cuts;
  int live = 0;
  for (int p = 0; p <= numGroups - 2; ++p) {
    live += delta[p];
    if (live == 1) cuts.push_back(p);
  }

  // Enforce the minimum partition size in original ops (Orbit's delta).
  std::vector<int> prefixOps(numGroups + 1, 0);
  for (int gi = 0; gi < numGroups; ++gi) {
    prefixOps[gi + 1] = prefixOps[gi] + grouping.groups[gi].members.size();
  }
  int minSize = std::max(1, partitionMinSize);
  SmallVector<int> kept;
  int begin = 0;
  for (int p : cuts) {
    if (prefixOps[p + 1] - prefixOps[begin] >= minSize) {
      kept.push_back(p);
      begin = p + 1;
    }
  }
  while (!kept.empty() &&
         prefixOps[numGroups] - prefixOps[kept.back() + 1] < minSize) {
    kept.pop_back();
  }

  begin = 0;
  for (int p : kept) {
    partitions.push_back(
        {begin, p + 1, cutValueAt(p), prefixOps[p + 1] - prefixOps[begin]});
    begin = p + 1;
  }
  partitions.push_back(
      {begin, numGroups, Value(), prefixOps[numGroups] - prefixOps[begin]});
  return partitions;
}

LogicalResult ILPBootstrapPlacementAnalysis::solve() {
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) return failure();
  Block* body = genericOp.getBody();

  OpGrouping grouping = computeOpGrouping(body, solver, options.compress);
  SmallVector<std::pair<Value, ValueState>, 4> argStates;
  if (failed(computeArgInitStates(body, solver, options, argStates)))
    return failure();
  SmallVector<Partition> partitions =
      computePartitions(grouping, body, solver, options.partitionMinSize);

  LLVM_DEBUG({
    int numOps = 0;
    for (const OpGroup& group : grouping.groups) numOps += group.members.size();
    llvm::dbgs() << "ilp-bootstrap-placement: " << numOps << " tracked ops in "
                 << grouping.groups.size() << " groups across "
                 << partitions.size() << " partitions\n";
  });

  auto adoptSolution = [&](PartitionSolution& soln) {
    for (auto& management : soln.nodeManagement)
      nodeManagement.push_back(management);
    for (auto& management : soln.edgeManagement)
      edgeManagement.push_back(management);
    for (auto& [op, useBootstrap] : soln.bootstrapDecisions)
      solution.insert(std::make_pair(op, useBootstrap));
    for (auto& [value, level] : soln.levelBefore)
      solutionLevelBeforeBootstrap.insert(std::make_pair(value, level));
    for (auto& [value, level] : soln.levelAfter)
      solutionLevelAfterBootstrap.insert(std::make_pair(value, level));
  };

  if (partitions.size() == 1) {
    PartitionBoundary boundary;
    boundary.pinnedInputs = argStates;
    boundary.applyYieldConstraints = true;
    boundary.sizeInOps = partitions[0].sizeInOps;
    std::optional<PartitionSolution> soln;
    if (failed(solvePartition(body, solver, options, grouping, 0,
                              grouping.groups.size(), boundary, soln)))
      return failure();
    if (!soln.has_value()) {
      llvm::errs() << "No feasible solution found (the problem may be "
                      "infeasible).\n";
      return failure();
    }
    adoptSolution(*soln);
    return success();
  }

  // Dynamic program over partitions (Orbit's QBP + DP stitch): each partition
  // is solved for every reachable boundary input state and every enumerated
  // boundary output level; the realized output scale keys the next
  // partition's input states.
  using StateKey = std::pair<int, int>;
  struct DpEntry {
    double totalCost;
    StateKey prevState;
    int solutionIndex;
  };
  // Sentinel input for the first partition (arguments are pinned separately)
  // and sentinel output for the last.
  const StateKey kNoState(-1, -1);
  std::map<StateKey, DpEntry> prevTable;
  prevTable.insert({kNoState, {0.0, kNoState, -1}});
  std::vector<std::vector<PartitionSolution>> solutions(partitions.size());
  std::vector<std::map<StateKey, DpEntry>> tables(partitions.size());

  for (size_t k = 0; k < partitions.size(); ++k) {
    const Partition& partition = partitions[k];
    bool isLast = k == partitions.size() - 1;
    std::map<StateKey, DpEntry> table;

    for (auto& [inState, prevEntry] : prevTable) {
      PartitionBoundary boundary;
      if (k == 0) {
        boundary.pinnedInputs = argStates;
      } else {
        boundary.pinnedInputs.push_back(
            {partitions[k - 1].cutValue, {inState.first, inState.second}});
      }
      boundary.applyYieldConstraints = isLast;
      boundary.sizeInOps = partition.sizeInOps;

      SmallVector<int> outLevels;
      if (isLast) {
        outLevels.push_back(-1);
      } else {
        boundary.outputValue = partition.cutValue;
        for (int level = 0; level <= options.bootstrapWaterline; ++level)
          outLevels.push_back(level);
      }

      for (int outLevel : outLevels) {
        boundary.outputLevel = outLevel;
        std::optional<PartitionSolution> soln;
        if (failed(solvePartition(body, solver, options, grouping,
                                  partition.groupBegin, partition.groupEnd,
                                  boundary, soln)))
          return failure();
        LLVM_DEBUG(llvm::dbgs()
                   << "partition " << k << " in(" << inState.first << ","
                   << inState.second << ") outLvl " << outLevel << ": "
                   << (soln.has_value()
                           ? ("cost " + std::to_string(soln->cost) + " out(" +
                              std::to_string(soln->outState.level) + "," +
                              std::to_string(soln->outState.scale) + ")")
                           : std::string("infeasible"))
                   << "\n");
        if (!soln.has_value()) continue;

        StateKey outKey =
            isLast ? kNoState
                   : StateKey(soln->outState.level, soln->outState.scale);
        double totalCost = prevEntry.totalCost + soln->cost;
        auto it = table.find(outKey);
        if (it != table.end() && it->second.totalCost <= totalCost) continue;
        soln->inKey = inState;
        int solutionIndex = solutions[k].size();
        solutions[k].push_back(std::move(*soln));
        table[outKey] = {totalCost, inState, solutionIndex};
      }
    }

    if (table.empty()) {
      genericOp->emitError()
          << "no feasible boundary state found for ILP partition " << k;
      return failure();
    }

    // Prune to at most one boundary scale per boundary level: keep the
    // minimal scale whose cost is within tolerance of the level's best.
    if (!isLast) {
      double tolerance =
          0.8 * partition.sizeInOps * options.costModel.rescaleCost;
      std::map<int, double> bestCostPerLevel;
      for (auto& [key, entry] : table) {
        auto it = bestCostPerLevel.find(key.first);
        if (it == bestCostPerLevel.end() || entry.totalCost < it->second)
          bestCostPerLevel[key.first] = entry.totalCost;
      }
      std::map<StateKey, DpEntry> pruned;
      std::map<int, bool> doneLevel;
      for (auto& [key, entry] : table) {
        // std::map iterates scales in increasing order per level, so the
        // first qualifying scale for a level wins.
        if (doneLevel[key.first]) continue;
        if (entry.totalCost <= bestCostPerLevel[key.first] + tolerance) {
          pruned.insert({key, entry});
          doneLevel[key.first] = true;
        }
      }
      table = std::move(pruned);
    }

    tables[k] = table;
    prevTable = std::move(table);
  }

  // Backtrack from the final sentinel state and stitch the chosen solutions.
  StateKey key = kNoState;
  SmallVector<int> chosen(partitions.size());
  for (int k = partitions.size() - 1; k >= 0; --k) {
    const DpEntry& entry = tables[k].at(key);
    chosen[k] = entry.solutionIndex;
    key = entry.prevState;
  }
  for (size_t k = 0; k < partitions.size(); ++k) {
    adoptSolution(solutions[k][chosen[k]]);
  }
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
  os << "bootstrap waterline: " << options.bootstrapWaterline << "\n";
  os << "scale waterline: " << options.scaleWaterline << "\n";
  os << "scale factor bits: " << options.scaleFactorBits << "\n\n";

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
