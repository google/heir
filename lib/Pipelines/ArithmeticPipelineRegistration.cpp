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
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"         // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

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
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }
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

    // Add client interface
    // TODO(#891): Extend add client interface to schemes other than BGV.
    switch (scheme) {
      case RLWEScheme::bgvScheme: {
        auto addClientInterfaceOptions = lwe::AddClientInterfaceOptions{};
        // OpenFHE's pke API, which this pipeline generates, is always
        // public-key
        addClientInterfaceOptions.usePublicKey = true;
        addClientInterfaceOptions.oneValuePerHelperFn = true;
        pm.addPass(lwe::createAddClientInterface(addClientInterfaceOptions));
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
    auto configureCryptoContextOptions =
        openfhe::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    pm.addPass(
        openfhe::createConfigureCryptoContext(configureCryptoContextOptions));
  };
}

}  // namespace mlir::heir
