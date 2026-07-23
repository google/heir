#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_

#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
class DataFlowSolver;
namespace heir {

struct InsertMgmtPipelineOptions {
  bool includeFloats;
  bool modReduceAfterMul;
  bool modReduceBeforeMulIncludeFirstMul;
  std::optional<int64_t> bootstrapWaterline;
  int64_t levelBudget;
};

// Run the secret-insert-mgmt pipeline.
//
// If includeFloats is true, then patterns will be included for arith.*f ops
// instead of just arith.*i ops.
//
// If bootstrapWaterline is present, a step of the pipeline will include a
// waterline bootstrap insertion routine.
LogicalResult runInsertMgmtPipeline(Operation* top,
                                    const InsertMgmtPipelineOptions& options);

void insertMgmtInitForPlaintexts(Operation* top, bool includeFloats);

void insertModReduceBeforeOrAfterMult(Operation* top, bool afterMul,
                                      bool beforeMulIncludeFirstMul,
                                      bool includeFloats);

void insertRelinearizeAfterMult(Operation* top, bool includeFloats);

void adjustLevelsForRegionBranchOps(Operation* top);

void handleCrossLevelOps(Operation* top, int* idCounter, bool includeFloats);

void handleCrossMulDepthOps(Operation* top, int* idCounter, bool includeFloats);

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline,
                              int levelBudget, bool includeFloats,
                              int* idCounter);

/// Peels the first iteration of loops if they have plaintext initial values
/// and secret yielded values. This is needed to ensure level analysis can
/// see the level growth correctly.
void peelPlaintextIterations(Operation* top);

/// Inserts bootstraps for loop iter args that are secret to ensure level
/// invariance across iterations. Applies to the given loop operation.
void bootstrapLoopIterArgs(Operation* loopOp, DataFlowSolver* solver);

/// Inserts mgmt.init for plaintext branch terminators and level reduce ops
/// to ensure level invariance across region branches.
void makeRegionBranchOpsLevelInvariant(Operation* top);

/// Returns a list of loops that are not level invariant, and hence require
/// bootstrap insertion and may benefit from level unrolling. The returned
/// vector is ordered to ensure that nested loops appear before their parent
/// loops. This implies that the unrollLoopForLevelUtilization may attempt to
/// unroll the loops returned by this function in a forward-iteration order.
SmallVector<Operation*> getNonInvariantLoops(Operation* top,
                                             DataFlowSolver* solver);

/// Unrolls the given loop operation for level utilization.
void unrollLoopForLevelUtilization(Operation* loopOp, DataFlowSolver* solver,
                                   int levelBudget);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_
