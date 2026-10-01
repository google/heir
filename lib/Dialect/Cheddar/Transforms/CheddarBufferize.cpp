#include "lib/Dialect/Cheddar/Transforms/CheddarBufferize.h"

#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

void buildCheddarBufferizationPipeline(OpPassManager& pm) {
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  bufferization::OneShotBufferizePassOptions oneShot;
  oneShot.bufferizeFunctionBoundaries = true;
  oneShot.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShot));
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  bufferization::DropEquivalentBufferResultsPassOptions dropEquivalent;
  dropEquivalent.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createDropEquivalentBufferResultsPass(dropEquivalent));
  bufferization::BufferResultsToOutParamsPassOptions outParams;
  outParams.hoistStaticAllocs = true;
  outParams.addResultAttribute = true;
  outParams.modifyPublicFunctions = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParams));
  pm.addPass(createCanonicalizerPass());
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
