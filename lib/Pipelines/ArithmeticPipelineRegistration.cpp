#include "lib/Pipelines/ArithmeticPipelineRegistration.h"

#include <cstdlib>
#include <string>

#include "lib/Dialect/BGV/Conversions/BGVToOpenfhe/BGVToOpenfhe.h"
#include "lib/Dialect/CKKS/Conversions/CKKSToOpenfhe/CKKSToOpenfhe.h"
#include "lib/Dialect/LWE/Transforms/AddClientInterface.h"
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"
#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "lib/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"
#include "lib/Dialect/TensorExt/Transforms/InsertRotate.h"
#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"
#include "lib/Pipelines/PipelineRegistration.h"
#include "lib/Transforms/ApplyFolders/ApplyFolders.h"
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/MemrefToArith/MemrefToArith.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

using mlir::func::FuncOp;

namespace mlir::heir {

void heirSIMDVectorizerPipelineBuilder(OpPassManager &manager) {
  // For now we unroll loops to enable insert-rotate, but we would like to be
  // smarter about this and do an affine loop analysis.
  // TODO(#589): avoid unrolling loops
  manager.addPass(createFullLoopUnroll());

  // These two passes are required in this position for a relatively nuanced
  // reason. insert-rotate doesn't have general match support. In particular,
  // if a tensor extract from a secret is combined with a tensor extract from a
  // constant 2D tensor (e.g., the weight matrix of a convolution), then
  // insert-rotate won't be able to tell the difference and understand that the
  // extracted value from the 2D tensor should be splatted.
  //
  // Canonicalize supports folding these away, but is too slow to run on the
  // unrolled loop. Instead, this "empty" pass uses the greedy rewrite engine
  // to apply folding patterns, including for tensor.extract, which converts a
  // constant weight matrix into the underlying arith.constant values, which
  // are supported as a splattable non-tensor input in insert-rotate. Then the
  // canonicalize pass can be run efficiently to achieve the same effect as if
  // the canonicalize pass were run alone.
  manager.addPass(createApplyFolders());
  manager.addPass(createCanonicalizerPass());

  // Insert rotations aligned to slot targets. Future work should provide
  // alternative methods to optimally align rotations, and allow the user to
  // configure this via pipeline options.
  manager.addPass(tensor_ext::createInsertRotate());
  manager.addPass(createCSEPass());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());

  manager.addPass(tensor_ext::createCollapseInsertionChains());
  manager.addPass(createSCCPPass());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());

  manager.addPass(tensor_ext::createRotateAndReduce());
  manager.addPass(createSCCPPass());
  manager.addPass(createApplyFolders());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
}

void mlirToSecretArithmeticPipelineBuilder(OpPassManager &pm) {
  pm.addPass(createWrapGeneric());
  convertToDataObliviousPipelineBuilder(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Apply linalg kernels
  pm.addPass(createLinalgCanonicalizations());
  pm.addPass(heir::linalg::createLinalgToTensorExt());

  // Vectorize and optimize rotations
  heirSIMDVectorizerPipelineBuilder(pm);

  // Balance Operations
  pm.addPass(createOperationBalancer());
}

void tosaToArithPipelineBuilder(OpPassManager &pm) {
  // TOSA to linalg
  ::mlir::heir::tosaToLinalg(pm);

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
  pm.addPass(affine::createAffineParallelizePass());
  pm.addPass(createFullLoopUnroll());
  pm.addPass(createForwardStoreToLoad());
  pm.addNestedPass<FuncOp>(createRemoveUnusedMemRef());

  // Cleanup
  pm.addPass(createMemrefGlobalReplacePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

void mlirToRLWEPipeline(OpPassManager &pm,
                        const MlirToRLWEPipelineOptions &options,
                        const RLWEScheme scheme) {
  mlirToSecretArithmeticPipelineBuilder(pm);

  // Prepare to lower to RLWE Scheme
  pm.addPass(secret::createSecretDistributeGeneric());
  pm.addPass(createCanonicalizerPass());

  // Lower to RLWE Scheme
  switch (scheme) {
    case RLWEScheme::ckksScheme: {
      auto secretToCKKSOpts = SecretToCKKSOptions{};
      secretToCKKSOpts.polyModDegree = options.ciphertextDegree;
      pm.addPass(createSecretToCKKS(secretToCKKSOpts));
      break;
    }
    case RLWEScheme::bgvScheme: {
      auto secretToBGVOpts = SecretToBGVOptions{};
      secretToBGVOpts.polyModDegree = options.ciphertextDegree;
      pm.addPass(createSecretToBGV(secretToBGVOpts));
      pm.addPass(createOptimizeRelinearization());
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  // Add client interface (helper functions)
  auto addClientInterfaceOptions = lwe::AddClientInterfaceOptions{};
  addClientInterfaceOptions.usePublicKey = options.usePublicKey;
  addClientInterfaceOptions.oneValuePerHelperFn = options.oneValuePerHelperFn;
  pm.addPass(lwe::createAddClientInterface(addClientInterfaceOptions));

  // TODO (#1145): This should also generate keygen/param gen functions,
  // which can then be lowered to backend specific stuff later.
}

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(const RLWEScheme scheme) {
  return [=](OpPassManager &pm, const MlirToRLWEPipelineOptions &options) {
    mlirToRLWEPipeline(pm, options, scheme);
  };
}

RLWEPipelineBuilder mlirToOpenFheRLWEPipelineBuilder(const RLWEScheme scheme) {
  return [=](OpPassManager &pm, const MlirToRLWEPipelineOptions &options) {
    // lower to RLWE scheme
    mlirToRLWEPipeline(pm, options, scheme);

    // Convert to OpenFHE
    switch (scheme) {
      case RLWEScheme::bgvScheme: {
        pm.addPass(bgv::createBGVToOpenfhe());
        break;
      }
      case RLWEScheme::ckksScheme: {
        pm.addPass(ckks::createCKKSToOpenfhe());
        break;
      }
      default:
        llvm::errs() << "Unsupported RLWE scheme: " << scheme;
        exit(EXIT_FAILURE);
    }

    pm.addPass(createCanonicalizerPass());
    // TODO (#1145): OpenFHE context configuration should NOT do its own
    // analysis but instead use information put into the IR by previous passes
    auto configureCryptoContextOptions =
        openfhe::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    pm.addPass(
        openfhe::createConfigureCryptoContext(configureCryptoContextOptions));
  };
}

void registerTosaToArithPipeline() {
  PassPipelineRegistration<>(
      "tosa-to-arith", "Arithmetic modules to arith tfhe-rs pipeline.",
      [](OpPassManager &pm) { tosaToArithPipelineBuilder(pm); });
}

}  // namespace mlir::heir
