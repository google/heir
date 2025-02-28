#include "lib/Pipelines/ArithmeticPipelineRegistration.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "lib/Dialect/BGV/Conversions/BGVToLWE/BGVToLWE.h"
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/Transforms/AddClientInterface.h"
#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"
#include "lib/Dialect/Lattigo/Transforms/AllocToInplace.h"
#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/CountAddAndKeySwitch.h"
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"
#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "lib/Dialect/Secret/Transforms/MergeAdjacentGenerics.h"
#include "lib/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"
#include "lib/Dialect/TensorExt/Transforms/InsertRotate.h"
#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"
#include "lib/Pipelines/PipelineRegistration.h"
#include "lib/Transforms/ApplyFolders/ApplyFolders.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/ValidateNoise/ValidateNoise.h"
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"    // from @llvm-project

namespace mlir::heir {

void heirSIMDVectorizerPipelineBuilder(OpPassManager &manager,
                                       bool disableLoopUnroll) {
  // For now we unroll loops to enable insert-rotate, but we would like to be
  // smarter about this and do an affine loop analysis.
  // TODO(#589): avoid unrolling loops
  if (!disableLoopUnroll) {
    manager.addPass(createFullLoopUnroll());
  }

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

void mlirToSecretArithmeticPipelineBuilder(
    OpPassManager &pm, const MlirToRLWEPipelineOptions &options) {
  pm.addPass(createWrapGeneric());
  convertToDataObliviousPipelineBuilder(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Apply linalg kernels
  // Linalg canonicalization
  // TODO(#1191): enable dropping unit dims to convert matmul to matvec/vecmat
  // pm.addPass(createDropUnitDims());
  pm.addPass(createLinalgCanonicalizations());
  // Layout assignment and lowering
  // TODO(#1191): enable layout propagation after implementing the rest
  // of the layout lowering pipeline.
  // pm.addPass(createLayoutPropagation());
  // Note: LinalgToTensorExt requires that linalg.matmuls are the only operation
  // within a secret.generic. This is to ensure that any tensor type conversions
  // (padding a rectangular matrix to a square diagonalized matrix) can be
  // performed without any type mismatches.
  std::vector<std::string> opsToDistribute = {"linalg.matmul"};
  auto distributeOpts = secret::SecretDistributeGenericOptions{
      .opsToDistribute = llvm::to_vector(opsToDistribute)};
  pm.addPass(createSecretDistributeGeneric(distributeOpts));
  pm.addPass(createCanonicalizerPass());
  auto linalgToTensorExtOptions = linalg::LinalgToTensorExtOptions{};
  linalgToTensorExtOptions.tilingSize = options.ciphertextDegree;
  pm.addPass(heir::linalg::createLinalgToTensorExt(linalgToTensorExtOptions));
  pm.addPass(secret::createSecretMergeAdjacentGenerics());

  // Vectorize and optimize rotations
  heirSIMDVectorizerPipelineBuilder(pm, options.experimentalDisableLoopUnroll);

  // Balance Operations
  pm.addPass(createOperationBalancer());
}

void mlirToRLWEPipeline(OpPassManager &pm,
                        const MlirToRLWEPipelineOptions &options,
                        const RLWEScheme scheme) {
  mlirToSecretArithmeticPipelineBuilder(pm, options);

  // place mgmt.op and MgmtAttr for BGV
  // which is required for secret-to-<scheme> lowering
  switch (scheme) {
    case RLWEScheme::bgvScheme: {
      auto secretInsertMgmtBGVOptions = SecretInsertMgmtBGVOptions{};
      secretInsertMgmtBGVOptions.includeFirstMul =
          options.modulusSwitchBeforeFirstMul;
      pm.addPass(createSecretInsertMgmtBGV(secretInsertMgmtBGVOptions));
      break;
    }
    case RLWEScheme::bfvScheme: {
      pm.addPass(createSecretInsertMgmtBFV());
      break;
    }
    case RLWEScheme::ckksScheme: {
      auto secretInsertMgmtCKKSOptions = SecretInsertMgmtCKKSOptions{};
      secretInsertMgmtCKKSOptions.includeFirstMul =
          options.modulusSwitchBeforeFirstMul;
      secretInsertMgmtCKKSOptions.slotNumber = options.ciphertextDegree;
      pm.addPass(createSecretInsertMgmtCKKS(secretInsertMgmtCKKSOptions));
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  // Optimize relinearization at mgmt dialect level
  pm.addPass(createOptimizeRelinearization());

  // IR is stable now, compute scheme param
  switch (scheme) {
    case RLWEScheme::bgvScheme: {
      auto generateParamOptions = GenerateParamBGVOptions{};
      if (!options.noiseModel.empty()) {
        generateParamOptions.model = options.noiseModel;
      }
      generateParamOptions.plaintextModulus = options.plaintextModulus;
      generateParamOptions.slotNumber = options.ciphertextDegree;
      pm.addPass(createGenerateParamBGV(generateParamOptions));

      auto validateNoiseOptions = ValidateNoiseOptions{};
      if (!options.noiseModel.empty()) {
        validateNoiseOptions.model = options.noiseModel;
      }
      validateNoiseOptions.annotateNoiseBound = options.annotateNoiseBound;
      pm.addPass(createValidateNoise(validateNoiseOptions));
      break;
    }
    case RLWEScheme::bfvScheme: {
      auto generateParamOptions = GenerateParamBFVOptions{};
      generateParamOptions.plaintextModulus = options.plaintextModulus;
      generateParamOptions.slotNumber = options.ciphertextDegree;
      pm.addPass(createGenerateParamBFV(generateParamOptions));
      break;
    }
    case RLWEScheme::ckksScheme: {
      auto generateParamOptions = GenerateParamCKKSOptions{};
      generateParamOptions.firstModBits = options.firstModBits;
      generateParamOptions.scalingModBits = options.scalingModBits;
      generateParamOptions.slotNumber = options.ciphertextDegree;
      pm.addPass(createGenerateParamCKKS(generateParamOptions));
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  if (scheme == RLWEScheme::bgvScheme) {
    // count add and keyswitch for Openfhe
    // this pass only works for BGV now
    pm.addPass(openfhe::createCountAddAndKeySwitch());
  }

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
    case RLWEScheme::bgvScheme:
    case RLWEScheme::bfvScheme: {
      auto secretToBGVOpts = SecretToBGVOptions{};
      secretToBGVOpts.polyModDegree = options.ciphertextDegree;
      pm.addPass(createSecretToBGV(secretToBGVOpts));
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  // Add client interface (helper functions)
  auto addClientInterfaceOptions = lwe::AddClientInterfaceOptions{};
  addClientInterfaceOptions.usePublicKey = options.usePublicKey;
  pm.addPass(lwe::createAddClientInterface(addClientInterfaceOptions));

  // TODO (#1145): This should also generate keygen/param gen functions,
  // which can then be lowered to backend specific stuff later.
}

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(const RLWEScheme scheme) {
  return [=](OpPassManager &pm, const MlirToRLWEPipelineOptions &options) {
    mlirToRLWEPipeline(pm, options, scheme);
  };
}

BackendPipelineBuilder toOpenFhePipelineBuilder() {
  return [=](OpPassManager &pm, const BackendOptions &options) {
    // Convert the common trivial subset of CKKS/BGV to LWE
    pm.addPass(bgv::createBGVToLWE());
    pm.addPass(ckks::createCKKSToLWE());

    // insert debug handler calls
    if (options.debug) {
      lwe::AddDebugPortOptions addDebugPortOptions;
      addDebugPortOptions.entryFunction = options.entryFunction;
      pm.addPass(lwe::createAddDebugPort(addDebugPortOptions));
    }

    // Convert LWE (and scheme-specific CKKS/BGV ops) to OpenFHE
    pm.addPass(lwe::createLWEToOpenfhe());

    // Simplify, in case the lowering revealed redundancy
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // TODO (#1145): OpenFHE context configuration should NOT do its own
    // analysis but instead use information put into the IR by previous passes
    auto configureCryptoContextOptions =
        openfhe::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    pm.addPass(
        openfhe::createConfigureCryptoContext(configureCryptoContextOptions));
  };
}

BackendPipelineBuilder toLattigoPipelineBuilder() {
  return [=](OpPassManager &pm, const BackendOptions &options) {
    // Convert to (common trivial subset of) LWE
    // TODO (#1193): Replace `--bgv-to-lwe` with `--bgv-common-to-lwe`
    pm.addPass(bgv::createBGVToLWE());
    pm.addPass(ckks::createCKKSToLWE());

    // insert debug handler calls
    if (options.debug) {
      lwe::AddDebugPortOptions addDebugPortOptions;
      addDebugPortOptions.entryFunction = options.entryFunction;
      pm.addPass(lwe::createAddDebugPort(addDebugPortOptions));
    }

    // Convert LWE (and scheme-specific BGV ops) to Lattigo
    pm.addPass(lwe::createLWEToLattigo());

    // Convert Alloc Ops to Inplace Ops
    pm.addPass(lattigo::createAllocToInplace());

    // Simplify, in case the lowering revealed redundancy
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    auto configureCryptoContextOptions =
        lattigo::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    pm.addPass(
        lattigo::createConfigureCryptoContext(configureCryptoContextOptions));
  };
}
}  // namespace mlir::heir
