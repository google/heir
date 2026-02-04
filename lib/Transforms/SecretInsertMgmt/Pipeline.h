#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_

#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
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

void handleCrossLevelOps(Operation* top, int* idCounter, bool includeFloats);

void handleCrossMulDepthOps(Operation* top, int* idCounter, bool includeFloats);

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline);

void makeLoopsTypeAndLevelInvariant(Operation* top);

void unrollLoopsForLevelUtilization(Operation* top, int levelBudget);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_PIPELINE_H_
