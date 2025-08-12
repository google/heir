#include "lib/Dialect/Mgmt/IR/MgmtOps.h"

#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Dialect/Mgmt/IR/MgmtCanonicalization.cpp.inc"

void ModReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeModReduce>(context);
}

void LevelReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<MergeLevelReduce>(context);
  results.add<ModReduceAfterLevelReduce>(context);
  results.add<AdjustScaleAfterLevelReduce>(context);
}

void AdjustScaleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<ModReduceAfterAdjustScale>(context);
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

void cleanupInitOp(Operation* top) {
  top->walk([&](mgmt::InitOp initOp) {
    if (initOp->use_empty()) initOp.erase();
  });
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
