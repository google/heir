#include "lib/Dialect/Mgmt/IR/MgmtOps.h"

#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Dialect/Mgmt/IR/MgmtCanonicalization.cpp.inc"

void ModReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<MergeModReduce>(context);
}

void LevelReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<MergeLevelReduce>(context);
  results.add<ModReduceAfterLevelReduce>(context);
  results.add<AdjustScaleAfterLevelReduce>(context);
}

void AdjustScaleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<ModReduceAfterAdjustScale>(context);
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
