#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

struct InsertMgmtPipelineOptions {
  bool includeFloats;
  bool modReduceAfterMul;
  bool modReduceBeforeMulIncludeFirstMul;
  std::optional<int64_t> bootstrapWaterline;
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

void rerunDataflow(DataFlowSolver& solver, Operation* top);

void insertMgmtInitForPlaintexts(Operation* top, DataFlowSolver& solver,
                                 bool includeFloats);

void insertModReduceBeforeOrAfterMult(Operation* top, DataFlowSolver& solver,
                                      bool afterMul,
                                      bool beforeMulIncludeFirstMul,
                                      bool includeFloats);

void insertRelinearizeAfterMult(Operation* top, DataFlowSolver& solver,
                                bool includeFloats);

void handleCrossLevelOps(Operation* top, DataFlowSolver& solver, int* idCounter,
                         bool includeFloats);

void handleCrossMulDepthOps(Operation* top, DataFlowSolver& solver,
                            int* idCounter, bool includeFloats);

void insertBootstrapWaterLine(Operation* top, DataFlowSolver& solver,
                              int bootstrapWaterline);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_
