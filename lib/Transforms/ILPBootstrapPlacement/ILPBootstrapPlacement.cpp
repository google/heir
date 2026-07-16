#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h"

#include <optional>
#include <string>
#include <utility>

#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/JSON.h"                // from @llvm-project
#include "llvm/include/llvm/Support/MemoryBuffer.h"        // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "ilp-bootstrap-placement"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ILPBOOTSTRAPPLACEMENT
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h.inc"

// Read one op's per-level latency array (index i holds the latency at level
// i + 1).
static std::optional<SmallVector<double>> readLatencies(
    const llvm::json::Object& latencyTable, llvm::StringRef opName) {
  const llvm::json::Array* latencies = latencyTable.getArray(opName);
  if (!latencies) return std::nullopt;

  SmallVector<double> values;
  for (const llvm::json::Value& latencyValue : *latencies) {
    std::optional<double> latency = latencyValue.getAsNumber();
    if (!latency) continue;
    values.push_back(*latency);
  }
  if (values.empty()) return std::nullopt;
  return values;
}

static std::optional<double> averagePositiveLatency(
    const llvm::json::Object& latencyTable, llvm::StringRef opName) {
  auto values = readLatencies(latencyTable, opName);
  if (!values) return std::nullopt;
  double sum = 0;
  int count = 0;
  for (double value : *values) {
    if (value <= 0) continue;
    sum += value;
    ++count;
  }
  if (count == 0) return std::nullopt;
  return sum / count;
}

static std::optional<double> maxPositiveLatency(
    const llvm::json::Object& latencyTable, llvm::StringRef opName) {
  auto values = readLatencies(latencyTable, opName);
  if (!values) return std::nullopt;
  double maxValue = *llvm::max_element(*values);
  if (maxValue <= 0) return std::nullopt;
  return maxValue;
}

// Least-squares fit of cost(level) = slope * level + intercept over a
// per-level latency array, where array index i holds the latency at level
// i + 1.
static std::optional<LinearCost> fitLinearCost(
    const llvm::json::Object& latencyTable, llvm::StringRef opName) {
  auto values = readLatencies(latencyTable, opName);
  if (!values || values->empty()) return std::nullopt;

  int n = values->size();
  if (n == 1) return LinearCost{0.0, (*values)[0]};

  double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (int i = 0; i < n; ++i) {
    double x = i + 1;
    double y = (*values)[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  double intercept = (sumY - slope * sumX) / n;
  return LinearCost{slope, intercept};
}

static FailureOr<OpCostModel> loadOrbitCostModel(llvm::StringRef path) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError) return failure();

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }

  const llvm::json::Object* root = parsed->getAsObject();
  if (!root) return failure();
  const llvm::json::Object* latencyTable = root->getObject("latencyTable");
  if (!latencyTable) return failure();

  // Bootstrap and rescale enter the objective as constant per-decision costs:
  // bootstrap is the average of positive samples (levels below the bootstrap
  // range are typically recorded as zero) and rescale is the per-level
  // maximum.
  std::optional<double> bootstrapCost =
      averagePositiveLatency(*latencyTable, "bootstrap");
  std::optional<double> rescaleCost =
      maxPositiveLatency(*latencyTable, "rescale");
  if (!bootstrapCost || !rescaleCost) return failure();

  OpCostModel costModel;
  costModel.bootstrapCost = *bootstrapCost;
  costModel.rescaleCost = *rescaleCost;

  struct LevelCostKey {
    llvm::StringRef name;
    LinearCost* target;
  };
  LevelCostKey levelCostKeys[] = {
      {"addCtCt", &costModel.addCtCt}, {"addCtPt", &costModel.addCtPt},
      {"mulCtCt", &costModel.mulCtCt}, {"mulCtPt", &costModel.mulCtPt},
      {"rotate", &costModel.rotate},   {"negate", &costModel.negate},
  };
  for (auto& [key, target] : levelCostKeys) {
    std::optional<LinearCost> fitted = fitLinearCost(*latencyTable, key);
    if (!fitted) return failure();
    *target = *fitted;
  }
  costModel.hasLevelCosts = true;

  return costModel;
}

struct ILPBootstrapPlacement
    : impl::ILPBootstrapPlacementBase<ILPBootstrapPlacement> {
  using ILPBootstrapPlacementBase::ILPBootstrapPlacementBase;

  struct ScaleConfig {
    bool isCKKS;
    int scaleWaterline;
    int scaleFactorBits;

    bool levelOnly() const { return !isCKKS; }
    ILPBootstrapPlacementAnalysis::ScaleMode analysisScaleMode() const {
      return isCKKS ? ILPBootstrapPlacementAnalysis::ScaleMode::kCKKS
                    : ILPBootstrapPlacementAnalysis::ScaleMode::kLevelOnly;
    }
  };

  bool hasCKKSAttrs(Operation* op) {
    return op->hasAttr("ckks.schemeParam") || op->hasAttr("scheme.ckks");
  }

  bool moduleTargetsCKKS(Operation* module) {
    if (hasCKKSAttrs(module)) return true;
    bool found = false;
    module->walk([&](Operation* op) {
      if (hasCKKSAttrs(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  }

  ScaleConfig getScaleConfig(Operation* module) {
    bool isCKKS = moduleTargetsCKKS(module);
    int effectiveScaleWaterline = scaleWaterline;
    int effectiveScaleFactorBits =
        isCKKS ? scaleFactorBits : effectiveScaleWaterline;
    return {isCKKS, effectiveScaleWaterline, effectiveScaleFactorBits};
  }

  void eraseExistingPlacementMgmtOps(secret::GenericOp genericOp) {
    SmallVector<Operation*> opsToErase;
    genericOp->walk([&](Operation* op) {
      if (isa<mgmt::RelinearizeOp, mgmt::ModReduceOp, mgmt::LevelReduceOp,
              mgmt::AdjustScaleOp, mgmt::BootstrapOp>(op)) {
        opsToErase.push_back(op);
      }
    });
    for (Operation* op : opsToErase) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
  }

  LogicalResult processSecretGenericOp(
      secret::GenericOp genericOp, DataFlowSolver* solver,
      const ScaleConfig& scaleConfig,
      SmallVector<ILPBootstrapPlacementAnalysis::NodeManagement, 32>*
          nodeManagement,
      SmallVector<ILPBootstrapPlacementAnalysis::EdgeManagement, 32>*
          edgeManagement) {
    OpCostModel effectiveCostModel;
    effectiveCostModel.bootstrapCost = bootstrapCost;
    effectiveCostModel.rescaleCost = rescaleCost;
    if (!orbitCostModel.empty()) {
      FailureOr<OpCostModel> loadedCostModel =
          loadOrbitCostModel(orbitCostModel);
      if (failed(loadedCostModel)) {
        llvm::errs() << "failed to load Orbit cost model from `"
                     << orbitCostModel << "`\n";
        genericOp->emitError() << "failed to load Orbit cost model from `"
                               << orbitCostModel << "`";
        return failure();
      }
      effectiveCostModel = *loadedCostModel;
    }

    ILPBootstrapPlacementAnalysis analysis(
        genericOp, solver, bootstrapWaterline, scaleConfig.scaleWaterline,
        scaleConfig.scaleFactorBits, bootstrapLevelLowerBound,
        effectiveCostModel, scaleConfig.analysisScaleMode());
    if (failed(analysis.solve())) {
      genericOp->emitError(
          "Failed to solve the bootstrap placement optimization problem");
      return failure();
    }
    LLVM_DEBUG(analysis.printSolution(llvm::dbgs()));
    for (auto placement : analysis.getNodeManagement())
      nodeManagement->push_back(placement);
    for (auto placement : analysis.getEdgeManagement())
      edgeManagement->push_back(placement);
    return success();
  }

  std::pair<Value, Operation*> followMgmtChain(Value value) {
    Value chainValue = value;
    Operation* chainEnd = value.getDefiningOp();
    while (chainValue.hasOneUse()) {
      Operation* user = *chainValue.getUsers().begin();
      if (isa<mgmt::RelinearizeOp, mgmt::ModReduceOp, mgmt::LevelReduceOp,
              mgmt::AdjustScaleOp, mgmt::BootstrapOp>(user)) {
        chainValue = user->getResult(0);
        chainEnd = user;
        continue;
      }
      break;
    }
    return {chainValue, chainEnd};
  }

  int ceilDivPositive(int numerator, int denominator) {
    if (numerator <= 0) return 0;
    return (numerator + denominator - 1) / denominator;
  }

  Value insertAfterCurrent(OpBuilder& b, Value current, Operation*& insertAfter,
                           Value original, Operation* newOp) {
    current.replaceAllUsesExcept(newOp->getResult(0), newOp);
    if (current == original) {
      original.replaceAllUsesExcept(newOp->getResult(0), newOp);
    }
    insertAfter = newOp;
    return newOp->getResult(0);
  }

  FailureOr<Value> decodeDirectTransitionAfter(OpBuilder& b, Value value,
                                               Operation*& insertAfter,
                                               int inputLevel, int inputScale,
                                               int outputLevel, int outputScale,
                                               const ScaleConfig& scaleConfig) {
    Value current = value;
    int currentLevel = inputLevel;
    int currentScale = inputScale;
    int scaleMax = scaleConfig.scaleFactorBits + 2 * scaleConfig.scaleWaterline;
    bool levelOnly = scaleConfig.levelOnly();

    int rescaleCount = levelOnly ? 0
                                 : ceilDivPositive(currentScale - outputScale,
                                                   scaleConfig.scaleFactorBits);
    if (outputLevel + rescaleCount < currentLevel) {
      int levelToDrop = currentLevel - outputLevel - rescaleCount;
      b.setInsertionPointAfter(insertAfter);
      auto levelReduceOp = mgmt::LevelReduceOp::create(
          b, insertAfter->getLoc(), current, b.getI64IntegerAttr(levelToDrop));
      current = insertAfterCurrent(b, current, insertAfter, value,
                                   levelReduceOp.getOperation());
      currentLevel -= levelToDrop;
    }

    for (int i = rescaleCount - 1; i >= 0; --i) {
      int targetScale = std::min(scaleMax - scaleConfig.scaleFactorBits,
                                 outputScale + i * scaleConfig.scaleFactorBits);
      if (currentScale < targetScale + scaleConfig.scaleFactorBits) {
        b.setInsertionPointAfter(insertAfter);
        auto adjustScaleOp = mgmt::AdjustScaleOp::create(
            b, insertAfter->getLoc(), current,
            b.getI64IntegerAttr(adjustScaleIdCounter++));
        current = insertAfterCurrent(b, current, insertAfter, value,
                                     adjustScaleOp.getOperation());
        currentScale = targetScale + scaleConfig.scaleFactorBits;
      }

      b.setInsertionPointAfter(insertAfter);
      auto modReduceOp =
          mgmt::ModReduceOp::create(b, insertAfter->getLoc(), current);
      current = insertAfterCurrent(b, current, insertAfter, value,
                                   modReduceOp.getOperation());
      --currentLevel;
      currentScale -= scaleConfig.scaleFactorBits;
    }

    if (!levelOnly && currentScale < outputScale) {
      b.setInsertionPointAfter(insertAfter);
      auto adjustScaleOp = mgmt::AdjustScaleOp::create(
          b, insertAfter->getLoc(), current,
          b.getI64IntegerAttr(adjustScaleIdCounter++));
      current = insertAfterCurrent(b, current, insertAfter, value,
                                   adjustScaleOp.getOperation());
      currentScale = outputScale;
    }

    if (currentLevel != outputLevel || currentScale != outputScale) {
      insertAfter->emitError()
          << "failed to decode Orbit node transition from (" << inputLevel
          << ", " << inputScale << ") to (" << outputLevel << ", "
          << outputScale << ")";
      return failure();
    }
    return current;
  }

  FailureOr<Value> decodeDirectTransitionBefore(
      OpBuilder& b, Operation* op, Value current, int inputLevel,
      int inputScale, int outputLevel, int outputScale,
      const ScaleConfig& scaleConfig) {
    int currentLevel = inputLevel;
    int currentScale = inputScale;
    int scaleMax = scaleConfig.scaleFactorBits + 2 * scaleConfig.scaleWaterline;
    bool levelOnly = scaleConfig.levelOnly();

    int rescaleCount = levelOnly ? 0
                                 : ceilDivPositive(currentScale - outputScale,
                                                   scaleConfig.scaleFactorBits);
    if (outputLevel + rescaleCount < currentLevel) {
      int levelToDrop = currentLevel - outputLevel - rescaleCount;
      b.setInsertionPoint(op);
      current = mgmt::LevelReduceOp::create(b, op->getLoc(), current,
                                            b.getI64IntegerAttr(levelToDrop));
      currentLevel -= levelToDrop;
    }

    for (int i = rescaleCount - 1; i >= 0; --i) {
      int targetScale = std::min(scaleMax - scaleConfig.scaleFactorBits,
                                 outputScale + i * scaleConfig.scaleFactorBits);
      if (currentScale < targetScale + scaleConfig.scaleFactorBits) {
        b.setInsertionPoint(op);
        current = mgmt::AdjustScaleOp::create(
            b, op->getLoc(), current,
            b.getI64IntegerAttr(adjustScaleIdCounter++));
        currentScale = targetScale + scaleConfig.scaleFactorBits;
      }
      b.setInsertionPoint(op);
      current = mgmt::ModReduceOp::create(b, op->getLoc(), current);
      --currentLevel;
      currentScale -= scaleConfig.scaleFactorBits;
    }

    if (!levelOnly && currentScale < outputScale) {
      b.setInsertionPoint(op);
      current = mgmt::AdjustScaleOp::create(
          b, op->getLoc(), current,
          b.getI64IntegerAttr(adjustScaleIdCounter++));
      currentScale = outputScale;
    }

    if (currentLevel != outputLevel || currentScale != outputScale) {
      op->emitError() << "failed to decode Orbit edge transition from ("
                      << inputLevel << ", " << inputScale << ") to ("
                      << outputLevel << ", " << outputScale << ")";
      return failure();
    }
    return current;
  }

  FailureOr<Value> decodeNodeTransition(
      const ILPBootstrapPlacementAnalysis::NodeManagement& placement,
      const ScaleConfig& scaleConfig) {
    auto [current, insertAfter] = followMgmtChain(placement.value);
    if (!insertAfter) return failure();

    OpBuilder b(&getContext());
    if (!placement.useBootstrap) {
      return decodeDirectTransitionAfter(
          b, current, insertAfter, placement.inputLevel, placement.inputScale,
          placement.outputLevel, placement.outputScale, scaleConfig);
    }

    FailureOr<Value> beforeBootstrap = decodeDirectTransitionAfter(
        b, current, insertAfter, placement.inputLevel, placement.inputScale,
        bootstrapLevelLowerBound, scaleConfig.scaleFactorBits, scaleConfig);
    if (failed(beforeBootstrap)) return failure();

    int postBootstrapRescales =
        scaleConfig.levelOnly() ? 0
                                : ceilDivPositive(scaleConfig.scaleFactorBits -
                                                      placement.outputScale,
                                                  scaleConfig.scaleFactorBits);
    int bootstrapTargetLevel = placement.outputLevel + postBootstrapRescales;
    if (bootstrapTargetLevel > bootstrapWaterline) {
      insertAfter->emitError()
          << "decoded bootstrap target level " << bootstrapTargetLevel
          << " exceeds bootstrap waterline " << bootstrapWaterline;
      return failure();
    }

    b.setInsertionPointAfter(insertAfter);
    auto bootstrapOp =
        mgmt::BootstrapOp::create(b, insertAfter->getLoc(), *beforeBootstrap);
    Value afterBootstrap =
        insertAfterCurrent(b, *beforeBootstrap, insertAfter, placement.value,
                           bootstrapOp.getOperation());

    int currentLevel = bootstrapWaterline;
    if (bootstrapTargetLevel < bootstrapWaterline) {
      int levelToDrop = bootstrapWaterline - bootstrapTargetLevel;
      b.setInsertionPointAfter(insertAfter);
      auto levelReduceOp =
          mgmt::LevelReduceOp::create(b, insertAfter->getLoc(), afterBootstrap,
                                      b.getI64IntegerAttr(levelToDrop));
      afterBootstrap =
          insertAfterCurrent(b, afterBootstrap, insertAfter, placement.value,
                             levelReduceOp.getOperation());
      currentLevel = bootstrapTargetLevel;
    }

    return decodeDirectTransitionAfter(
        b, afterBootstrap, insertAfter, currentLevel,
        scaleConfig.scaleFactorBits, placement.outputLevel,
        placement.outputScale, scaleConfig);
  }

  LogicalResult decodeEdgeTransition(
      const ILPBootstrapPlacementAnalysis::EdgeManagement& placement,
      const ScaleConfig& scaleConfig) {
    Operation* op = placement.op;
    if (!op || placement.operandNumber >= op->getNumOperands())
      return success();
    OpBuilder b(&getContext());
    Value current = op->getOperand(placement.operandNumber);
    FailureOr<Value> managed = decodeDirectTransitionBefore(
        b, op, current, placement.inputLevel, placement.inputScale,
        placement.outputLevel, placement.outputScale, scaleConfig);
    if (failed(managed)) return failure();
    op->setOperand(placement.operandNumber, *managed);
    return success();
  }

  void runOnOperation() override {
    Operation* module = getOperation();

    SmallVector<secret::GenericOp> genericOps;
    module->walk(
        [&](secret::GenericOp genericOp) { genericOps.push_back(genericOp); });
    for (secret::GenericOp genericOp : genericOps) {
      eraseExistingPlacementMgmtOps(genericOp);
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    ScaleConfig scaleConfig = getScaleConfig(module);

    SmallVector<ILPBootstrapPlacementAnalysis::NodeManagement, 32>
        nodeManagement;
    SmallVector<ILPBootstrapPlacementAnalysis::EdgeManagement, 32>
        edgeManagement;
    auto result = module->walk([&](secret::GenericOp genericOp) {
      if (failed(processSecretGenericOp(genericOp, &solver, scaleConfig,
                                        &nodeManagement, &edgeManagement)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Relinearize after every mul.
    insertRelinearizeAfterMult(getOperation(), /*includeFloats=*/true);

    for (const auto& placement : nodeManagement) {
      if (failed(decodeNodeTransition(placement, scaleConfig))) {
        signalPassFailure();
        return;
      }
    }

    for (const auto& placement : edgeManagement) {
      if (failed(decodeEdgeTransition(placement, scaleConfig))) {
        signalPassFailure();
        return;
      }
    }

    OpPassManager nested("builtin.module");
    nested.addPass(createCanonicalizerPass());
    nested.addPass(createCSEPass());
    nested.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(nested, getOperation()))) {
      signalPassFailure();
      return;
    }
  }

  int64_t adjustScaleIdCounter = 0;
};

}  // namespace heir
}  // namespace mlir
