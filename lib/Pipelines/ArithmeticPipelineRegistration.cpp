#include "lib/Pipelines/ArithmeticPipelineRegistration.h"

#include <cstdlib>
#include <string>

#include "lib/Dialect/BGV/Conversions/BGVToLWE/BGVToLWE.h"
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"
#include "lib/Dialect/Debug/Transforms/Passes.h"
#include "lib/Dialect/Debug/Transforms/ValidateNames.h"
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"
#include "lib/Dialect/Lattigo/Transforms/AllocToInPlace.h"
#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/AllocToInPlace.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/CountAddAndKeySwitch.h"
#include "lib/Dialect/Openfhe/Transforms/FastRotationPrecompute.h"
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/Transforms/AddDebugPort.h"
#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "lib/Dialect/Secret/Transforms/ImportExecutionResult.h"
#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "lib/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Dialect/TensorExt/Transforms/InsertRotate.h"
#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"
#include "lib/Pipelines/PipelineRegistration.h"
#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h"
#include "lib/Transforms/AddClientInterface/AddClientInterface.h"
#include "lib/Transforms/ApplyFolders/ApplyFolders.h"
#include "lib/Transforms/BooleanVectorizer/BooleanVectorizer.h"
#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"
#include "lib/Transforms/DropUnitDims/DropUnitDims.h"
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"
#include "lib/Transforms/FoldConstantTensors/FoldConstantTensors.h"
#include "lib/Transforms/FoldPlaintextMasks/FoldPlaintextMasks.h"
#include "lib/Transforms/ForwardInsertSliceToExtractSlice/ForwardInsertSliceToExtractSlice.h"
#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "lib/Transforms/InlineActivations/InlineActivations.h"
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h"
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/SelectRewrite/SelectRewrite.h"
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h"
#include "lib/Transforms/ValidateNoise/ValidateNoise.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir::heir {

void heirSIMDVectorizerPipelineBuilder(OpPassManager& manager,
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

void lowerAssignLayout(OpPassManager& pm, bool unroll = false) {
  // Lower linalg.generics produced by ConvertToCiphertextSemantics
  // (assign_layout lowering) to affine loops.
  pm.addPass(createTensorLinalgToAffineLoops());
  pm.addNestedPass<func::FuncOp>(affine::createAffineExpandIndexOpsPass());
  pm.addNestedPass<func::FuncOp>(affine::createSimplifyAffineStructuresPass());
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addNestedPass<func::FuncOp>(createForwardInsertSliceToExtractSlice());

  // The lowered assign_layout ops involve plaintext operations that are still
  // inside secret.generic, and are not handled well by downstream noise models
  // and parameter selection passes. Canonicalize to hoist them out of
  // secret.generic.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // TODO(#1181): remove the need to loop unroll
  if (unroll) {
    pm.addPass(createFullLoopUnroll());
  }
}

// Implement layout conversions as shift networks
void implementShiftNetworkPipelineBuilder(OpPassManager& pm) {
  pm.addPass(tensor_ext::createImplementShiftNetwork());
  // implement shift networks produces some naive repeated plaintext masks

  // CSE in prep for folding plaintext masks
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Clean up foldable repeated masks
  pm.addPass(createFoldPlaintextMasks());

  // The cleaned up masks may enable further simplifications
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlirToSecretArithmeticPipelineBuilder(
    OpPassManager& pm, const MlirToRLWEPipelineOptions& options) {
  pm.addPass(debug::createDebugValidateNames());
  pm.addPass(createWrapGeneric());
  convertToDataObliviousPipelineBuilder(pm);
  pm.addPass(createSelectRewrite());
  pm.addPass(createCompareToSignRewrite());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Vectorize and optimize rotations
  // TODO(#2320): figure out where this fits in the new pipeline
  heirSIMDVectorizerPipelineBuilder(pm, options.experimentalDisableLoopUnroll);
  mathToPolynomialApproximationBuilder(pm);

  // Layout assignment and optimization
  LayoutPropagationOptions layoutPropagationOptions;
  layoutPropagationOptions.ciphertextSize = options.ciphertextDegree;
  pm.addPass(createLayoutPropagation(layoutPropagationOptions));
  pm.addPass(createLayoutOptimization());
  // Layout conversions may be repeated, so run CSE
  pm.addPass(createCSEPass());

  // Linalg kernel implementation
  ConvertToCiphertextSemanticsOptions convertToCiphertextSemanticsOptions;
  convertToCiphertextSemanticsOptions.ciphertextSize = options.ciphertextDegree;
  pm.addPass(
      createConvertToCiphertextSemantics(convertToCiphertextSemanticsOptions));

  pm.addPass(createApplyFolders());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(tensor_ext::createImplementRotateAndReduce());

  implementShiftNetworkPipelineBuilder(pm);

  // Balance Operations
  pm.addPass(createOperationBalancer());

  lowerAssignLayout(pm, false);

  // Add encrypt/decrypt helper functions for each function argument and return
  // value.
  AddClientInterfaceOptions addClientInterfaceOptions;
  addClientInterfaceOptions.ciphertextSize = options.ciphertextDegree;
  pm.addPass(createAddClientInterface(addClientInterfaceOptions));

  // Clean up after lowering assign_layout and various related packing code
  pm.addPass(createApplyFolders());
  pm.addPass(createFoldConstantTensors());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlirToPlaintextPipelineBuilder(OpPassManager& pm,
                                    const PlaintextBackendOptions& options) {
  pm.addPass(debug::createDebugValidateNames());
  linalgPreprocessingBuilder(pm);

  // Convert to secret arithmetic
  MlirToRLWEPipelineOptions mlirToRLWEPipelineOptions;
  mlirToRLWEPipelineOptions.ciphertextDegree = options.plaintextSize;
  mlirToSecretArithmeticPipelineBuilder(pm, mlirToRLWEPipelineOptions);

  if (options.debug) {
    // Insert debug handler calls
    pm.addPass(secret::createSecretAddDebugPort(
        secret::SecretAddDebugPortOptions{.insertDebugAfterEveryOp = true}));
  } else {
    pm.addPass(secret::createSecretAddDebugPort(
        secret::SecretAddDebugPortOptions{.insertDebugAfterEveryOp = false}));
  }

  pm.addPass(secret::createSecretDistributeGeneric());
  pm.addPass(createCanonicalizerPass());

  mod_arith::SecretToModArithOptions secretToModArithOptions;
  secretToModArithOptions.plaintextModulus = options.plaintextModulus;
  pm.addPass(createSecretToModArith(secretToModArithOptions));
  lowerAssignLayout(pm, false);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Convert to standard dialect
  pm.addPass(tensor_ext::createTensorExtToTensor());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  polynomialToLLVMPipelineBuilder(pm);
}

void mlirToRLWEPipeline(OpPassManager& pm,
                        const MlirToRLWEPipelineOptions& options,
                        const RLWEScheme scheme) {
  pm.addPass(debug::createDebugValidateNames());

  if (options.enableArithmetization) {
    mlirToSecretArithmeticPipelineBuilder(pm, options);
  } else {
    // Replicate the non-arithmetization related parts of the pipeline
    pm.addPass(createWrapGeneric());
    AddClientInterfaceOptions addClientInterfaceOptions;
    addClientInterfaceOptions.ciphertextSize = options.ciphertextDegree;
    addClientInterfaceOptions.enableLayoutAssignment = false;
    pm.addPass(createAddClientInterface(addClientInterfaceOptions));
  }

  pm.addPass(secret::createSecretAddDebugPort(secret::SecretAddDebugPortOptions{
      .insertDebugAfterEveryOp = options.debug}));

  // Only for debugging purpose.
  if (!options.plaintextExecutionResultFileName.empty()) {
    // Import execution result from file
    secret::SecretImportExecutionResultOptions
        secretImportExecutionResultOptions;
    secretImportExecutionResultOptions.fileName =
        options.plaintextExecutionResultFileName;
    pm.addPass(secret::createSecretImportExecutionResult(
        secretImportExecutionResultOptions));
  }

  // place mgmt.op and MgmtAttr for BGV
  // which is required for secret-to-<scheme> lowering
  switch (scheme) {
    case RLWEScheme::bgvScheme: {
      auto secretInsertMgmtBGVOptions = SecretInsertMgmtBGVOptions{};
      secretInsertMgmtBGVOptions.afterMul = options.modulusSwitchAfterMul;
      secretInsertMgmtBGVOptions.beforeMulIncludeFirstMul =
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
      secretInsertMgmtCKKSOptions.afterMul = options.modulusSwitchAfterMul;
      secretInsertMgmtCKKSOptions.beforeMulIncludeFirstMul =
          options.modulusSwitchBeforeFirstMul;
      secretInsertMgmtCKKSOptions.slotNumber = options.ciphertextDegree;
      secretInsertMgmtCKKSOptions.bootstrapWaterline =
          options.ckksBootstrapWaterline;
      secretInsertMgmtCKKSOptions.levelBudget = options.levelBudget;
      pm.addPass(createSecretInsertMgmtCKKS(secretInsertMgmtCKKSOptions));
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  // TODO(#2600): support loops in optimize-relinearization
  if (!options.experimentalDisableLoopUnroll) {
    OptimizeRelinearizationOptions optimizeRelinearizationOptions;
    optimizeRelinearizationOptions.allowMixedDegreeOperands = false;
    pm.addPass(createOptimizeRelinearization(optimizeRelinearizationOptions));
  }

  // IR is stable now

  // if we want to import execution result from file, propagate them to mgmt ops
  if (!options.plaintextExecutionResultFileName.empty()) {
    PropagateAnnotationOptions propagateAnnotationOptions;
    propagateAnnotationOptions.attrName =
        secret::SecretDialect::kArgExecutionResultAttrName;
    pm.addPass(createPropagateAnnotation(propagateAnnotationOptions));
  }

  // compute scheme param
  switch (scheme) {
    case RLWEScheme::bgvScheme: {
      auto generateParamOptions = GenerateParamBGVOptions{};
      if (!options.noiseModel.empty()) {
        generateParamOptions.model = options.noiseModel;
      }
      generateParamOptions.plaintextModulus = options.plaintextModulus;
      generateParamOptions.slotNumber = options.ciphertextDegree;
      generateParamOptions.usePublicKey = options.usePublicKey;
      generateParamOptions.encryptionTechniqueExtended =
          options.encryptionTechniqueExtended;
      pm.addPass(createGenerateParamBGV(generateParamOptions));

      auto validateNoiseOptions = ValidateNoiseOptions{};
      validateNoiseOptions.model = generateParamOptions.model;
      validateNoiseOptions.annotateNoiseBound = options.annotateNoiseBound;
      pm.addPass(createValidateNoise(validateNoiseOptions));

      pm.addPass(createPopulateScaleBGV());
      break;
    }
    case RLWEScheme::bfvScheme: {
      auto generateParamOptions = GenerateParamBFVOptions{};
      if (!options.noiseModel.empty()) {
        generateParamOptions.model = options.noiseModel;
      }
      generateParamOptions.modBits = options.bfvModBits;
      generateParamOptions.plaintextModulus = options.plaintextModulus;
      generateParamOptions.slotNumber = options.ciphertextDegree;
      generateParamOptions.usePublicKey = options.usePublicKey;
      generateParamOptions.encryptionTechniqueExtended =
          options.encryptionTechniqueExtended;
      pm.addPass(createGenerateParamBFV(generateParamOptions));

      auto validateNoiseOptions = ValidateNoiseOptions{};
      validateNoiseOptions.model = generateParamOptions.model;
      validateNoiseOptions.annotateNoiseBound = options.annotateNoiseBound;
      pm.addPass(createValidateNoise(validateNoiseOptions));

      // Fill the scale with 1 for correct Lattigo lowering
      pm.addPass(createPopulateScaleBGV());
      break;
    }
    case RLWEScheme::ckksScheme: {
      auto generateParamOptions = GenerateParamCKKSOptions{};
      generateParamOptions.firstModBits = options.firstModBits;
      generateParamOptions.scalingModBits = options.scalingModBits;
      generateParamOptions.slotNumber = options.ciphertextDegree;
      generateParamOptions.usePublicKey = options.usePublicKey;
      pm.addPass(createGenerateParamCKKS(generateParamOptions));

      PopulateScaleCKKSOptions populateScaleCKKSOptions;
      populateScaleCKKSOptions.beforeMulIncludeFirstMul =
          options.modulusSwitchBeforeFirstMul;
      pm.addPass(createPopulateScaleCKKS(populateScaleCKKSOptions));
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  if (scheme == RLWEScheme::bgvScheme || scheme == RLWEScheme::bfvScheme) {
    // count add and keyswitch for Openfhe
    // this pass only works for BGV/BFV
    pm.addPass(openfhe::createCountAddAndKeySwitch());
  }

  // Prepare to lower to RLWE Scheme
  pm.addPass(createCanonicalizerPass());
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

  pm.addPass(createForwardInsertToExtract());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());

  // Add a __preprocessed helper for offline pre-packing of plaintexts
  auto splitPreprocessingOptions = SplitPreprocessingOptions{};
  splitPreprocessingOptions.maxReturnValues = options.splitPreprocessing;
  pm.addPass(createSplitPreprocessing(splitPreprocessingOptions));

  ElementwiseToAffineOptions elementwiseOptions;
  elementwiseOptions.convertDialects = {"ckks", "bgv", "lwe"};
  pm.addPass(createElementwiseToAffine(elementwiseOptions));

  pm.addPass(tensor_ext::createTensorExtToTensor());
  lowerAssignLayout(pm, false);

  // TODO (#1145): This should also generate keygen/param gen functions,
  // which can then be lowered to backend specific stuff later.

  // At this point, due to the optimizations of implement-rotate-and-reduce,
  // the IR may still contain tensor_ext.rotate ops corresponding to rotations
  // of cleartexts. It also may not have been possible to fold them away
  // because many different rotations of the same plaintext are needed. In this
  // case, can just implement the rotations of the cleartexts directly in terms
  // of tensor ops, and they are already lazily encoded as plaintexts.
  pm.addPass(tensor_ext::createTensorExtToTensor());
}

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(const RLWEScheme scheme) {
  return [=](OpPassManager& pm, const MlirToRLWEPipelineOptions& options) {
    mlirToRLWEPipeline(pm, options, scheme);
  };
}

BackendPipelineBuilder toOpenFhePipelineBuilder() {
  return [=](OpPassManager& pm, const BackendOptions& options) {
    // Canonicalize to ensure the ciphertext operands are in the first operand
    // of ct-pt ops.
    pm.addPass(createCanonicalizerPass());

    // Convert the common trivial subset of CKKS/BGV to LWE
    pm.addPass(bgv::createBGVToLWE());
    pm.addPass(ckks::createCKKSToLWE());

    // insert debug handler calls
    lwe::AddDebugPortOptions addDebugPortOptions;
    addDebugPortOptions.entryFunction = options.entryFunction;
    addDebugPortOptions.insertDebugAfterEveryOp = options.debug;
    pm.addPass(lwe::createAddDebugPort(addDebugPortOptions));

    // Convert LWE (and scheme-specific CKKS/BGV ops) to OpenFHE
    pm.addPass(lwe::createLWEToOpenfhe());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // TODO (#1145): OpenFHE context configuration should NOT do its own
    // analysis but instead use information put into the IR by previous passes
    auto configureCryptoContextOptions =
        openfhe::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    pm.addPass(
        openfhe::createConfigureCryptoContext(configureCryptoContextOptions));

    pm.addPass(openfhe::createFastRotationPrecompute());
    // Vectorize any operations
    pm.addPass(createBooleanVectorizer());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(openfhe::createAllocToInPlace());
  };
}

BackendPipelineBuilder toLattigoPipelineBuilder() {
  return [=](OpPassManager& pm, const BackendOptions& options) {
    // Convert to (common trivial subset of) LWE
    // TODO (#1193): Replace `--bgv-to-lwe` with `--bgv-common-to-lwe`
    pm.addPass(bgv::createBGVToLWE());
    pm.addPass(ckks::createCKKSToLWE());

    // insert debug handler calls
    lwe::AddDebugPortOptions addDebugPortOptions;
    addDebugPortOptions.entryFunction = options.entryFunction;
    addDebugPortOptions.insertDebugAfterEveryOp = options.debug;
    pm.addPass(lwe::createAddDebugPort(addDebugPortOptions));

    // Convert LWE (and scheme-specific BGV ops) to Lattigo
    pm.addPass(lwe::createLWEToLattigo());

    // Convert Alloc Ops to InPlace Ops
    // TODO(#2635): Disable until this is fixed.
    // pm.addPass(lattigo::createAllocToInPlace());

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

void linalgPreprocessingBuilder(OpPassManager& manager) {
  manager.addPass(createInlineActivations());
  manager.addPass(createActivationCanonicalizations());
  manager.addPass(createLinalgCanonicalizations());
  manager.addPass(createDropUnitDims());
  manager.addPass(createFoldConstantTensors());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSymbolDCEPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createLinalgCanonicalizations());
}

void torchLinalgToCkksBuilder(OpPassManager& manager,
                              const TorchLinalgToCkksPipelineOptions& options) {
  manager.addPass(debug::createDebugValidateNames());
  linalgPreprocessingBuilder(manager);
  MlirToRLWEPipelineOptions suboptions;

  suboptions.enableArithmetization = true;
  suboptions.ciphertextDegree = options.ciphertextDegree;
  suboptions.ckksBootstrapWaterline = options.ckksBootstrapWaterline;
  suboptions.scalingModBits = options.scalingModBits;
  suboptions.firstModBits = options.firstModBits;
  suboptions.splitPreprocessing = options.splitPreprocessing;

  mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::ckksScheme)(manager,
                                                                suboptions);
}

}  // namespace mlir::heir
