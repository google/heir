#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h"

#include <cmath>
#include <optional>
#include <string>
#include <utility>

#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
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

struct OrbitCostModel {
  int bootstrapCost;
  int rescaleCost;
};

static std::optional<int> averagePositiveLatency(const llvm::json::Object& root,
                                                 llvm::StringRef opName) {
  const llvm::json::Object* latencyTable = root.getObject("latencyTable");
  if (!latencyTable) return std::nullopt;

  const llvm::json::Array* latencies = latencyTable->getArray(opName);
  if (!latencies) return std::nullopt;

  double sum = 0;
  int count = 0;
  for (const llvm::json::Value& latencyValue : *latencies) {
    std::optional<double> latency = latencyValue.getAsNumber();
    if (!latency || *latency <= 0) continue;
    sum += *latency;
    ++count;
  }
  if (count == 0) return std::nullopt;
  return static_cast<int>(std::llround(sum / count));
}

static FailureOr<OrbitCostModel> loadOrbitCostModel(llvm::StringRef path) {
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

  std::optional<int> parsedBootstrapCost =
      averagePositiveLatency(*root, "earth.bootstrap_single");
  std::optional<int> parsedRescaleCost =
      averagePositiveLatency(*root, "earth.rescale_single");
  if (!parsedBootstrapCost || !parsedRescaleCost) return failure();

  return OrbitCostModel{*parsedBootstrapCost, *parsedRescaleCost};
}

struct ILPBootstrapPlacement
    : impl::ILPBootstrapPlacementBase<ILPBootstrapPlacement> {
  using ILPBootstrapPlacementBase::ILPBootstrapPlacementBase;

  LogicalResult processSecretGenericOp(
      secret::GenericOp genericOp, DataFlowSolver* solver,
      SmallVector<Value, 32>* valuesToBootstrap,
      SmallVector<ILPBootstrapPlacementAnalysis::OutputLevelReduction, 32>*
          outputLevelReductions,
      SmallVector<ILPBootstrapPlacementAnalysis::OperandLevelReduction, 32>*
          operandLevelReductions) {
    genericOp->walk([&](mgmt::BootstrapOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    int effectiveBootstrapCost = bootstrapCost;
    int effectiveRescaleCost = rescaleCost;
    if (!orbitCostModel.empty()) {
      FailureOr<OrbitCostModel> loadedCostModel =
          loadOrbitCostModel(orbitCostModel);
      if (failed(loadedCostModel)) {
        llvm::errs() << "failed to load Orbit cost model from `"
                     << orbitCostModel << "`\n";
        genericOp->emitError() << "failed to load Orbit cost model from `"
                               << orbitCostModel << "`";
        return failure();
      }
      effectiveBootstrapCost = loadedCostModel->bootstrapCost;
      effectiveRescaleCost = loadedCostModel->rescaleCost;
    }

    ILPBootstrapPlacementAnalysis analysis(
        genericOp, solver, bootstrapWaterline, scaleWaterline, scaleFactorBits,
        bootstrapLevelLowerBound, effectiveBootstrapCost, effectiveRescaleCost,
        useOrbitCompression);
    if (failed(analysis.solve())) {
      genericOp->emitError(
          "Failed to solve the bootstrap placement optimization problem");
      return failure();
    }
    LLVM_DEBUG(analysis.printSolution(llvm::dbgs()));
    for (Value v : analysis.getValuesToBootstrap())
      valuesToBootstrap->push_back(v);
    for (auto reduction : analysis.getOutputLevelReductions())
      outputLevelReductions->push_back(reduction);
    for (auto reduction : analysis.getOperandLevelReductions())
      operandLevelReductions->push_back(reduction);
    return success();
  }

  std::pair<Value, Operation*> followRelinearizeModReduceChain(Value value) {
    Value chainValue = value;
    Operation* chainEnd = value.getDefiningOp();
    while (chainValue.hasOneUse()) {
      Operation* user = *chainValue.getUsers().begin();
      if (isa<mgmt::RelinearizeOp>(user) || isa<mgmt::ModReduceOp>(user)) {
        chainValue = user->getResult(0);
        chainEnd = user;
        continue;
      }
      break;
    }
    return {chainValue, chainEnd};
  }

  void insertOutputLevelReductions(
      ArrayRef<ILPBootstrapPlacementAnalysis::OutputLevelReduction>
          outputLevelReductions) {
    OpBuilder b(&getContext());
    for (auto reduction : outputLevelReductions) {
      auto [toReduce, insertAfter] =
          followRelinearizeModReduceChain(reduction.value);
      if (!insertAfter) continue;

      b.setInsertionPointAfter(insertAfter);
      auto levelReduceOp = mgmt::LevelReduceOp::create(
          b, insertAfter->getLoc(), toReduce,
          b.getI64IntegerAttr(reduction.levelToDrop));
      toReduce.replaceAllUsesExcept(levelReduceOp.getResult(), {levelReduceOp});
    }
  }

  void insertOperandLevelReductions(
      ArrayRef<ILPBootstrapPlacementAnalysis::OperandLevelReduction>
          operandLevelReductions) {
    OpBuilder b(&getContext());
    for (auto reduction : operandLevelReductions) {
      Operation* op = reduction.op;
      if (!op || reduction.operandNumber >= op->getNumOperands()) continue;

      Value operand = op->getOperand(reduction.operandNumber);
      b.setInsertionPoint(op);
      auto levelReduceOp = mgmt::LevelReduceOp::create(
          b, op->getLoc(), operand, b.getI64IntegerAttr(reduction.levelToDrop));
      op->setOperand(reduction.operandNumber, levelReduceOp.getResult());
    }
  }

  void insertBootstrapsForValues(ArrayRef<Value> valuesToBootstrap) {
    OpBuilder b(&getContext());
    for (Value v : valuesToBootstrap) {
      // After modreduce/relinearize we have mul -> relinearize -> modreduce.
      // Follow the chain so we bootstrap the modreduce result (correct level
      // refresh) and insert after it.
      auto [toBootstrap, insertAfter] = followRelinearizeModReduceChain(v);
      if (!insertAfter) continue;
      b.setInsertionPointAfter(insertAfter);
      auto bootstrapOp =
          mgmt::BootstrapOp::create(b, insertAfter->getLoc(), toBootstrap);
      toBootstrap.replaceAllUsesExcept(bootstrapOp.getResult(), {bootstrapOp});
    }
  }

  void runOnOperation() override {
    Operation* module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    SmallVector<Value, 32> valuesToBootstrap;
    SmallVector<ILPBootstrapPlacementAnalysis::OutputLevelReduction, 32>
        outputLevelReductions;
    SmallVector<ILPBootstrapPlacementAnalysis::OperandLevelReduction, 32>
        operandLevelReductions;
    auto result = module->walk([&](secret::GenericOp genericOp) {
      if (failed(processSecretGenericOp(genericOp, &solver, &valuesToBootstrap,
                                        &outputLevelReductions,
                                        &operandLevelReductions)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Insert per-use level reductions before consumers, matching Orbit-style
    // edge rescale placement.
    insertOperandLevelReductions(operandLevelReductions);

    // Modreduce after every mul.
    insertModReduceBeforeOrAfterMult(getOperation(), /*afterMul=*/true,
                                     /*beforeMulIncludeFirstMul=*/false,
                                     /*includeFloats=*/true);

    // Relinearize after every mul.
    insertRelinearizeAfterMult(getOperation(), /*includeFloats=*/true);

    // Insert shared producer-output level reductions after mul management and
    // before bootstraps, matching Orbit's node rescale decisions.
    insertOutputLevelReductions(outputLevelReductions);

    // Insert bootstraps at the Values the ILP chose. Values remain valid.
    insertBootstrapsForValues(valuesToBootstrap);

    OpPassManager nested("builtin.module");
    nested.addPass(createCanonicalizerPass());
    nested.addPass(createCSEPass());
    nested.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(nested, getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
