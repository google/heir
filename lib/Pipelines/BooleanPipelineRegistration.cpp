#include "lib/Pipelines/BooleanPipelineRegistration.h"

#include <memory>
#include <string>
#include <vector>

#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToJaxite/CGGIToJaxite.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRustBool/CGGIToTfheRustBool.h"
#include "lib/Dialect/CGGI/Transforms/BooleanVectorizer.h"
#include "lib/Dialect/Secret/Conversions/SecretToCGGI/SecretToCGGI.h"
#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "lib/Pipelines/PipelineRegistration.h"
#include "lib/Transforms/DropUnitDims/DropUnitDims.h"
#include "lib/Transforms/FoldConstantTensors/FoldConstantTensors.h"
#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h"
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/MemrefToArith/MemrefToArith.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h"
#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h"
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#endif

using namespace mlir;
using namespace heir;
using mlir::func::FuncOp;

namespace mlir::heir {

static std::vector<std::string> opsToDistribute = {"secret.separator"};
static std::vector<unsigned> bitWidths = {1, 2, 4, 8, 16};

#ifndef HEIR_NO_YOSYS
CGGIPipelineBuilder mlirToCGGIPipelineBuilder(const std::string& yosysFilesPath,
                                              const std::string& abcPath) {
  return [=](OpPassManager& pm, const MLIRToCGGIPipelineOptions& options) {
    mlirToCGGIPipeline(pm, options, yosysFilesPath, abcPath);
  };
}

void mlirToCGGIPipeline(OpPassManager& pm,
                        const MLIRToCGGIPipelineOptions& options,
                        const std::string& yosysFilesPath,
                        const std::string& abcPath) {
  // TOSA to linalg
  ::mlir::heir::tosaToLinalg(pm);

  // Linalg to Affine loops
  // pm.addPass(createDropUnitDims());
  pm.addPass(createFoldConstantTensors());
  pm.addPass(createLinalgCanonicalizations());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createTensorLinalgToAffineLoops());

  // Affine loop optimizations
  pm.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  pm.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addNestedPass<FuncOp>(
      affine::createLoopFusionPass(0, 0, true, affine::FusionMode::Greedy));
  pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addPass(affine::createAffineParallelize());
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCanonicalizerPass());

  if (options.dataType == Bool) {
    pm.addPass(createFullLoopUnroll());
  }

  // Cleanup after loop unroll
  ::mlir::arith::ArithIntRangeNarrowingOptions arithOps;
  arithOps.bitwidthsSupported = llvm::to_vector(bitWidths);
  // This pass also materializes known constants, so forwards arith.constant
  // tensor<...> with tensor.extracts.
  pm.addPass(::mlir::arith::createArithIntRangeNarrowing(arithOps));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLinalgCanonicalizations());
  pm.addPass(createForwardInsertToExtract());
  pm.addPass(createFoldConstantTensors());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createRemoveDeadValuesPass());

  // Booleanize
  auto distributeOpts = secret::SecretDistributeGenericOptions{
      .opsToDistribute = llvm::to_vector(opsToDistribute)};
  switch (options.dataType) {
    case Bool:
      // Wrap with secret.generic and then distribute-generic.
      pm.addPass(createWrapGeneric());
      pm.addPass(secret::createSecretDistributeGeneric(distributeOpts));
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath, options.abcFast,
                                      options.unrollFactor,
                                      options.useSubmodules, options.mode));
      // Cleanup
      pm.addPass(mlir::createCSEPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createFoldConstantTensors());
      pm.addPass(createSCCPPass());
      pm.addPass(createSymbolDCEPass());

      // Lower combinational circuit to CGGI
      pm.addPass(secret::createSecretDistributeGeneric());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createSecretToCGGI());

      break;
    case Integer:
      pm.addPass(arith::createArithToCGGI());
      break;
  }
  // Cleanup SecretToCGGI
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLinalgCanonicalizations());
  pm.addPass(createForwardInsertToExtract());
  pm.addPass(createFoldConstantTensors());
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
}
#else
CGGIPipelineBuilder mlirToCGGIPipelineBuilder() {
  return [=](OpPassManager& pm, const MLIRToCGGIPipelineOptions& options) {
    mlirToCGGIPipeline(pm, options);
  };
}

void mlirToCGGIPipeline(OpPassManager& pm,
                        const MLIRToCGGIPipelineOptions& options) {
  // Bufferize
  ::mlir::heir::oneShotBufferize(pm);

  // Affine
  pm.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());
  pm.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  pm.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  pm.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  pm.addPass(createExpandCopyPass());
  pm.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addPass(memref::createFoldMemRefAliasOpsPass());

  // Affine loop optimizations
  pm.addNestedPass<FuncOp>(
      affine::createLoopFusionPass(0, 0, true, affine::FusionMode::Greedy));
  pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addPass(createForwardStoreToLoad());
  pm.addPass(affine::createAffineParallelize());
  pm.addPass(createForwardStoreToLoad());
  pm.addNestedPass<FuncOp>(createRemoveUnusedMemRef());

  // Cleanup
  pm.addPass(createMemrefGlobalReplacePass());
  ::mlir::arith::ArithIntRangeNarrowingOptions arithOps;
  arithOps.bitwidthsSupported = llvm::to_vector(bitWidths);
  pm.addPass(::mlir::arith::createArithIntRangeNarrowing(arithOps));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());

  // Booleanize
  switch (options.dataType) {
    case Integer:
      pm.addPass(arith::createArithToCGGI());
      break;
    case Bool:
      llvm_unreachable("Booleanization is not supported without Yosys.");
  }

  pm.addPass(createForwardStoreToLoad());
  pm.addPass(createRemoveUnusedMemRef());
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
}
#endif

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
    pm.addPass(cggi::createBooleanVectorizer());
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

JaxiteBackendPipelineBuilder toJaxitePipelineBuilder() {
  return [=](OpPassManager& pm, const CGGIBackendOptions& options) {
    if (options.parallelism > 0) {
      pm.addPass(cggi::createBooleanVectorizer(
          cggi::BooleanVectorizerOptions{.parallelism = options.parallelism}));
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
