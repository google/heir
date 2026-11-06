#include "lib/Pipelines/BooleanPipelineRegistration.h"

#include "lib/Dialect/CGGI/Conversions/CGGIToJaxite/CGGIToJaxite.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToSCIFRBool/CGGIToSCIFRBool.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRustBool/CGGIToTfheRustBool.h"
#include "lib/Transforms/BooleanVectorizer/BooleanVectorizer.h"
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "lib/Transforms/MemrefToArith/MemrefToArith.h"
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

using namespace mlir;
using namespace heir;

namespace mlir::heir {

CGGIBackendPipelineBuilder toTfheRsPipelineBuilder() {
  return [=](OpPassManager& pm) {
    // CGGI to Tfhe-Rust exit dialect
    pm.addPass(createCGGIToTfheRust());
    // CSE must be run before canonicalizer, so that redundant ops are
    // cleared before the canonicalizer hoists TfheRust ops.
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Cleanup loads and stores
    pm.addPass(
        createExpandCopyPass(ExpandCopyPassOptions{.disableAffineLoop = true}));
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(createForwardStoreToLoad());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
  };
}

CGGIBackendPipelineBuilder toFptPipelineBuilder() {
  return [=](OpPassManager& pm) {
    // Vectorize CGGI operations
    pm.addPass(createBooleanVectorizer());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());

    // CGGI to Tfhe-Rust exit dialect
    pm.addPass(createCGGIToTfheRustBool());
    // CSE must be run before canonicalizer, so that redundant ops are
    // cleared before the canonicalizer hoists TfheRust ops.
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Cleanup loads and stores
    pm.addPass(
        createExpandCopyPass(ExpandCopyPassOptions{.disableAffineLoop = true}));
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(createForwardStoreToLoad());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
  };
}

CGGIBackendPipelineBuilder toCGGICornamiPipelineBuilder() {
  return [=](OpPassManager& pm) {
    // Vectorize CGGI operations
    pm.addPass(createBooleanVectorizer());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());

    // CGGI to SCIFRBool exit dialect
    pm.addPass(cornami::scifrbool::createCGGIToSCIFRBool());
    // CSE must be run before canonicalizer, so that redundant ops are
    // cleared before the canonicalizer hoists TfheRust ops.
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Cleanup loads and stores
    pm.addPass(
        createExpandCopyPass(ExpandCopyPassOptions{.disableAffineLoop = true}));
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(createForwardStoreToLoad());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
  };
}

JaxiteBackendPipelineBuilder toJaxitePipelineBuilder() {
  return [=](OpPassManager& pm, const CGGIBackendOptions& options) {
    if (options.parallelism > 0) {
      pm.addPass(createBooleanVectorizer(
          BooleanVectorizerOptions{.parallelism = options.parallelism}));
      pm.addPass(createCSEPass());
      pm.addPass(createRemoveDeadValuesPass());
    }

    // CGGI to Jaxite exit dialect
    pm.addPass(createCGGIToJaxite());
    // CSE must be run before canonicalizer, so that redundant ops are
    // cleared before the canonicalizer hoists TfheRust ops.
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Cleanup loads and stores
    pm.addPass(
        createExpandCopyPass(ExpandCopyPassOptions{.disableAffineLoop = true}));
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(createForwardStoreToLoad());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
  };
}

}  // namespace mlir::heir
