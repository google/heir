#include "lib/Pipelines/ArithmeticPipelineRegistration.h"

#include <cstdlib>
#include <string>

#include "lib/Dialect/BGV/Conversions/BGVToLWE/BGVToLWE.h"
#include "lib/Dialect/CKKS/Transforms/CKKSToLWE.h"
#include "lib/Dialect/Debug/Transforms/ValidateNames.h"
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"
#include "lib/Dialect/LWE/Transforms/AnnotatePlaintextLevel.h"
#include "lib/Dialect/LWE/Transforms/ImplementTrivialEncryptionAsAddition.h"
#include "lib/Dialect/Lattigo/Transforms/AllocToInPlace.h"
#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/AllocToInPlace.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/CountAddAndKeySwitch.h"
#include "lib/Dialect/Openfhe/Transforms/FastRotationPrecompute.h"
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToLattigo/PreprocessingToLattigo.h"
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToOpenfhe/PreprocessingToOpenfhe.h"
#include "lib/Dialect/Preprocessing/Transforms/ValidatePreprocessing.h"
#include "lib/Dialect/Rotom/Transforms/Passes.h"
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
#include "lib/Transforms/ExternalizeConstants/ExternalizeConstants.h"
#include "lib/Transforms/FoldConstantTensors/FoldConstantTensors.h"
#include "lib/Transforms/FoldPlaintextMasks/FoldPlaintextMasks.h"
#include "lib/Transforms/ForwardInsertSliceToExtractSlice/ForwardInsertSliceToExtractSlice.h"
#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h"
#include "lib/Transforms/InlineActivations/InlineActivations.h"
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h"
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "lib/Transforms/ReductionCanonicalizations/ReductionCanonicalizations.h"
#include "lib/Transforms/RemoveUnusedPureCall/RemoveUnusedPureCall.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/SelectRewrite/SelectRewrite.h"
#include "lib/Transforms/SoftmaxCanonicalizations/SoftmaxCanonicalizations.h"
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h"
#include "lib/Transforms/ValidateNoise/ValidateNoise.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"      // from @llvm-project

namespace mlir::heir {

llvm::cl::opt<std::string> extConstOutputDir(
    "ext-const-output-dir",
    llvm::cl::desc(
        "Directory to write the externalized constant binary files to."),
    llvm::cl::init(""));

llvm::cl::opt<std::string> extConstRuntimeLoadDir(
    "ext-const-runtime-load-dir",
    llvm::cl::desc("Directory path to use in the load_resource op in the "
                   "generated MLIR."),
    llvm::cl::init(""));

llvm::cl::opt<int> extConstThreshold(
    "ext-const-threshold",
    llvm::cl::desc("Minimum number of elements to externalize a constant."),
    llvm::cl::init(1024));

void hecoSIMDVectorizerPipelineBuilder(OpPassManager& manager,
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

void cleanupAfterLowerAssignLayout(OpPassManager& pm) {
  // Lower linalg.generics produced by ConvertToCiphertextSemantics
  // (assign_layout lowering) to affine loops.
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createTensorLinalgToAffineLoops());
  pm.addNestedPass<func::FuncOp>(affine::createAffineExpandIndexOpsPass());
  pm.addNestedPass<func::FuncOp>(affine::createSimplifyAffineStructuresPass());
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addNestedPass<func::FuncOp>(createForwardInsertSliceToExtractSlice());

  // Cleanup for various reasons:
  //
  // - The lowered assign_layout ops involve plaintext operations that are still
  //   inside secret.generic, and are not handled well by downstream noise
  //   models and parameter selection passes. Canonicalize to hoist them out of
  //   secret.generic.
  // - Preprocessing helpers may make copies of dense constants whose original
  //   instances are still present and not needed, but may be threaded through
  //   via a function call argument.
  // - Preprocessing helpers may be sccp-ed significantly.
  pm.addPass(createApplyFolders());
  pm.addPass(createFoldConstantTensors());
  pm.addPass(createSCCPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
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
  hecoSIMDVectorizerPipelineBuilder(pm, options.experimentalDisableLoopUnroll);
  mathToPolynomialApproximationBuilder(pm, options.useCompositeRelu);

  // Layout assignment and optimization
  LayoutPropagationOptions layoutPropagationOptions;
  layoutPropagationOptions.minSlotCount = options.minSlotCount;
  pm.addPass(createLayoutPropagation(layoutPropagationOptions));
  LayoutOptimizationOptions layoutOptimizationOptions;
  layoutOptimizationOptions.minSlotCount = options.minSlotCount;
  pm.addPass(createLayoutOptimization(layoutOptimizationOptions));
  // Layout conversions may be repeated, so run CSE
  pm.addPass(createCSEPass());

  EarlyBootstrapPlacementOptions earlyBootstrapOptions;
  earlyBootstrapOptions.levelBudget = options.greedyLevelBudget;
  earlyBootstrapOptions.bootstrapWaterline = options.greedyBootstrapWaterline;
  pm.addPass(createEarlyBootstrapPlacement(earlyBootstrapOptions));

  // Linalg kernel implementation
  ConvertToCiphertextSemanticsOptions convertToCiphertextSemanticsOptions;
  convertToCiphertextSemanticsOptions.minSlotCount = options.minSlotCount;
  convertToCiphertextSemanticsOptions.unrollKernels =
      !options.experimentalDisableLoopUnroll;
  convertToCiphertextSemanticsOptions.codegenStrategy = options.codegenStrategy;
  pm.addPass(
      createConvertToCiphertextSemantics(convertToCiphertextSemanticsOptions));

  pm.addPass(createApplyFolders());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(tensor_ext::createImplementRotateAndReduce());

  implementShiftNetworkPipelineBuilder(pm);

  // Balance Operations
  pm.addPass(createOperationBalancer());

  // Add encrypt/decrypt helper functions for each function argument and return
  // value.
  AddClientInterfaceOptions addClientInterfaceOptions;
  addClientInterfaceOptions.minSlotCount = options.minSlotCount;
  pm.addPass(createAddClientInterface(addClientInterfaceOptions));

  cleanupAfterLowerAssignLayout(pm);
}

void mlirToPlaintextPipelineBuilder(OpPassManager& pm,
                                    const PlaintextBackendOptions& options) {
  pm.addPass(debug::createDebugValidateNames());
  linalgPreprocessingBuilder(pm);

  // Convert to secret arithmetic
  MlirToRLWEPipelineOptions mlirToRLWEPipelineOptions;
  mlirToRLWEPipelineOptions.minSlotCount = options.plaintextSize;
  mlirToSecretArithmeticPipelineBuilder(pm, mlirToRLWEPipelineOptions);

  // Insert debug handler calls and/or lower debug.validate
  pm.addPass(secret::createSecretAddDebugPort(secret::SecretAddDebugPortOptions{
      .insertDebugAfterEveryOp = options.debug}));

  pm.addPass(secret::createSecretDistributeGeneric());
  pm.addPass(createCanonicalizerPass());

  mod_arith::SecretToModArithOptions secretToModArithOptions;
  secretToModArithOptions.plaintextModulus = options.plaintextModulus;
  pm.addPass(createSecretToModArith(secretToModArithOptions));

  cleanupAfterLowerAssignLayout(pm);

  // Convert to standard dialect
  pm.addPass(tensor_ext::createTensorExtToTensor());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  polynomialToLLVMPipelineBuilder(pm);
}

void mlirToRotomPlaintextPipelineBuilder(OpPassManager& pm,
                                         const RotomPlaintextOptions& options) {
  pm.addPass(debug::createDebugValidateNames());
  // The torch/CKKS front end: linalg preprocessing and polynomial
  // activations, so the program reaching the layout search is the same one
  // the CKKS pipeline sees.
  linalgPreprocessingBuilder(pm);
  pm.addPass(createWrapGeneric());
  convertToDataObliviousPipelineBuilder(pm);
  pm.addPass(createSelectRewrite());
  pm.addPass(createCompareToSignRewrite());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  hecoSIMDVectorizerPipelineBuilder(pm, /*disableLoopUnroll=*/false);
  mathToPolynomialApproximationBuilder(pm, /*useCompositeRelu=*/false);

  // Rotom: seed, search, outline, materialize, lower.
  pm.addPass(rotom::createNormalizeContractions());
  rotom::SeedLayoutOptions seedOptions;
  seedOptions.n = options.ciphertextSize;
  pm.addPass(rotom::createSeedLayout(seedOptions));
  pm.addPass(rotom::createLayoutAssignment());
  pm.addPass(rotom::createOutlineKernels());
  pm.addPass(rotom::createMaterializeTensorExtLayout());
  ConvertToCiphertextSemanticsOptions convertOptions;
  convertOptions.minSlotCount = options.ciphertextSize;
  convertOptions.unrollKernels = true;
  pm.addPass(createConvertToCiphertextSemantics(convertOptions));
  pm.addPass(createInlinerPass());
  pm.addPass(tensor_ext::createImplementShiftNetwork());

  // The client interface packs plaintext arguments and encrypts/decrypts the
  // secret ones from their logical shapes.
  AddClientInterfaceOptions clientOptions;
  clientOptions.minSlotCount = options.ciphertextSize;
  clientOptions.enableLayoutAssignment = true;
  pm.addPass(createAddClientInterface(clientOptions));
  cleanupAfterLowerAssignLayout(pm);

  // The plaintext backend tail.
  pm.addPass(secret::createSecretDistributeGeneric());
  pm.addPass(createCanonicalizerPass());
  mod_arith::SecretToModArithOptions secretToModArithOptions;
  secretToModArithOptions.plaintextModulus = options.plaintextModulus;
  pm.addPass(createSecretToModArith(secretToModArithOptions));
  cleanupAfterLowerAssignLayout(pm);
  pm.addPass(tensor_ext::createTensorExtToTensor());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  polynomialToLLVMPipelineBuilder(pm);
}

static void validateCiphertextManagementOptions(
    const MlirToRLWEPipelineOptions& options, const RLWEScheme scheme) {
  // Check if any greedy-specific options are non-default
  bool hasGreedyOptions = options.greedyModulusSwitchAfterMul != false ||
                          options.greedyModulusSwitchBeforeFirstMul != false ||
                          options.greedyLevelBudget != 10 ||
                          options.greedyBootstrapWaterline != 0;

  // Check if any orbit-specific options are non-default
  bool hasOrbitOptions =
      options.orbitBootstrapWaterline != 3 ||
      options.orbitScaleWaterline != 40 || options.orbitScaleFactorBits != 51 ||
      options.orbitBootstrapLevelLowerBound != 0 ||
      options.orbitCostModel != "" || options.orbitBootstrapCost != 69320650 ||
      options.orbitRescaleCost != 40988;

  // Validate style exclusivity
  if (options.ciphertextManagementStyle == CiphertextManagementStyle::greedy &&
      hasOrbitOptions) {
    llvm::errs() << "Error: orbit-* options cannot be used with "
                 << "--ciphertext-management-style=greedy\n"
                 << "Either switch to --ciphertext-management-style=orbit-ilp "
                 << "or use greedy-* options instead.\n";
    exit(EXIT_FAILURE);
  }

  if (options.ciphertextManagementStyle ==
          CiphertextManagementStyle::orbitIlp &&
      hasGreedyOptions) {
    llvm::errs() << "Error: greedy-* options cannot be used with "
                 << "--ciphertext-management-style=orbit-ilp\n"
                 << "Either switch to --ciphertext-management-style=greedy "
                 << "or use orbit-* options instead.\n";
    exit(EXIT_FAILURE);
  }

  // Validate scheme compatibility
  if (scheme == RLWEScheme::bfvScheme) {
    if (options.ciphertextManagementStyle ==
        CiphertextManagementStyle::orbitIlp) {
      llvm::errs() << "Error: orbit-ilp ciphertext management style is not "
                      "supported for BFV scheme\n"
                   << "BFV does not support bootstrap operations.\n";
      exit(EXIT_FAILURE);
    }

    if (options.greedyBootstrapWaterline != 0) {  // non-default
      llvm::errs() << "Error: --greedy-bootstrap-waterline is not supported "
                      "for BFV scheme\n"
                   << "BFV does not support bootstrap operations.\n";
      exit(EXIT_FAILURE);
    }
  }
}

void mlirToRLWEPipeline(OpPassManager& pm,
                        const MlirToRLWEPipelineOptions& options,
                        const RLWEScheme scheme) {
  // Validate ciphertext management options
  validateCiphertextManagementOptions(options, scheme);

  pm.addPass(debug::createDebugValidateNames());
  if (options.enableArithmetization) {
    mlirToSecretArithmeticPipelineBuilder(pm, options);
  } else {
    // Replicate the non-arithmetization related parts of the pipeline
    pm.addPass(createWrapGeneric());
    AddClientInterfaceOptions addClientInterfaceOptions;
    addClientInterfaceOptions.minSlotCount = options.minSlotCount;
    addClientInterfaceOptions.enableLayoutAssignment = false;
    pm.addPass(createAddClientInterface(addClientInterfaceOptions));
  }

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

  // place mgmt.op and MgmtAttr for BGV/CKKS
  // which is required for secret-to-<scheme> lowering
  switch (scheme) {
    case RLWEScheme::bgvScheme: {
      if (options.ciphertextManagementStyle ==
          CiphertextManagementStyle::orbitIlp) {
        // ILP-based management for BGV
        auto ilpOptions = ILPBootstrapPlacementOptions{};
        ilpOptions.bootstrapWaterline = options.orbitBootstrapWaterline;
        ilpOptions.scaleWaterline = options.orbitScaleWaterline;
        ilpOptions.scaleFactorBits = options.orbitScaleFactorBits;
        ilpOptions.bootstrapLevelLowerBound =
            options.orbitBootstrapLevelLowerBound;
        ilpOptions.orbitCostModel = options.orbitCostModel;
        ilpOptions.bootstrapCost = options.orbitBootstrapCost;
        ilpOptions.rescaleCost = options.orbitRescaleCost;
        pm.addPass(createILPBootstrapPlacement(ilpOptions));
      } else {
        // Greedy management for BGV
        auto secretInsertMgmtBGVOptions = SecretInsertMgmtBGVOptions{};
        secretInsertMgmtBGVOptions.afterMul =
            options.greedyModulusSwitchAfterMul;
        secretInsertMgmtBGVOptions.beforeMulIncludeFirstMul =
            options.greedyModulusSwitchBeforeFirstMul;
        secretInsertMgmtBGVOptions.levelBudget = options.greedyLevelBudget;
        pm.addPass(createSecretInsertMgmtBGV(secretInsertMgmtBGVOptions));
      }
      break;
    }
    case RLWEScheme::bfvScheme: {
      // BFV doesn't use bootstrap management currently
      pm.addPass(createSecretInsertMgmtBFV());
      break;
    }
    case RLWEScheme::ckksScheme: {
      if (options.ciphertextManagementStyle ==
          CiphertextManagementStyle::orbitIlp) {
        // ILP-based management for CKKS
        auto ilpOptions = ILPBootstrapPlacementOptions{};
        ilpOptions.bootstrapWaterline = options.orbitBootstrapWaterline;
        ilpOptions.scaleWaterline = options.orbitScaleWaterline;
        ilpOptions.scaleFactorBits = options.orbitScaleFactorBits;
        ilpOptions.bootstrapLevelLowerBound =
            options.orbitBootstrapLevelLowerBound;
        ilpOptions.orbitCostModel = options.orbitCostModel;
        ilpOptions.bootstrapCost = options.orbitBootstrapCost;
        ilpOptions.rescaleCost = options.orbitRescaleCost;
        pm.addPass(createILPBootstrapPlacement(ilpOptions));
      } else {
        // Greedy management for CKKS
        auto secretInsertMgmtCKKSOptions = SecretInsertMgmtCKKSOptions{};
        secretInsertMgmtCKKSOptions.afterMul =
            options.greedyModulusSwitchAfterMul;
        secretInsertMgmtCKKSOptions.beforeMulIncludeFirstMul =
            options.greedyModulusSwitchBeforeFirstMul;
        secretInsertMgmtCKKSOptions.minSlotCount = options.minSlotCount;
        secretInsertMgmtCKKSOptions.bootstrapWaterline =
            options.greedyBootstrapWaterline;
        secretInsertMgmtCKKSOptions.levelBudget = options.greedyLevelBudget;
        pm.addPass(createSecretInsertMgmtCKKS(secretInsertMgmtCKKSOptions));
      }
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
      generateParamOptions.minSlotCount = options.minSlotCount;
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
      generateParamOptions.minSlotCount = options.minSlotCount;
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
      generateParamOptions.minSlotCount = options.minSlotCount;
      generateParamOptions.usePublicKey = options.usePublicKey;
      pm.addPass(createGenerateParamCKKS(generateParamOptions));

      PopulateScaleCKKSOptions populateScaleCKKSOptions;
      populateScaleCKKSOptions.beforeMulIncludeFirstMul =
          options.greedyModulusSwitchBeforeFirstMul;
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
      secretToCKKSOpts.minSlotCount = options.minSlotCount;
      pm.addPass(createSecretToCKKS(secretToCKKSOpts));
      break;
    }
    case RLWEScheme::bgvScheme:
    case RLWEScheme::bfvScheme: {
      auto secretToBGVOpts = SecretToBGVOptions{};
      secretToBGVOpts.minSlotCount = options.minSlotCount;
      pm.addPass(createSecretToBGV(secretToBGVOpts));
      break;
    }
    default:
      llvm::errs() << "Unsupported RLWE scheme: " << scheme;
      exit(EXIT_FAILURE);
  }

  // Lower debug.validate ops to function calls with private key
  pm.addPass(lwe::createAddDebugPort(
      lwe::AddDebugPortOptions{.minSlotCount = (int)options.minSlotCount,
                               .insertDebugAfterEveryOp = options.debug}));

  pm.addPass(createForwardInsertToExtract());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());

  // TODO(#2554): skip this pass if the backend supports trivial encryption
  pm.addPass(lwe::createImplementTrivialEncryptionAsAddition());

  // Record the level each plaintext is used at, so backends can encode only the
  // limbs it needs. This must run after the CSE above, which merges identical
  // encode ops, and before split-preprocessing, which severs the use chain
  // from an encode op to the ciphertext op consuming it.
  pm.addPass(lwe::createAnnotatePlaintextLevel());

  // Add a __preprocessed helper for offline pre-packing of plaintexts
  if (options.enableSplitPreprocessing) {
    pm.addPass(createSplitPreprocessing());
    pm.addPass(preprocessing::createValidatePreprocessing());
  }

  ElementwiseToAffineOptions elementwiseOptions;
  elementwiseOptions.convertDialects = {"ckks", "bgv", "lwe", "kernel"};
  pm.addPass(createElementwiseToAffine(elementwiseOptions));

  pm.addPass(tensor_ext::createTensorExtToTensor());
  cleanupAfterLowerAssignLayout(pm);

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

    if (!extConstOutputDir.empty()) {
      ExternalizeConstantsOptions extConstOptions;
      extConstOptions.outputDir = extConstOutputDir;
      extConstOptions.runtimeLoadDir = extConstRuntimeLoadDir;
      extConstOptions.thresholdElements = extConstThreshold;
      pm.addPass(createExternalizeConstants(extConstOptions));
    }

    // insert debug handler calls
    lwe::AddDebugPortOptions addDebugPortOptions{
        .entryFunction = options.entryFunction,
        .insertDebugAfterEveryOp = options.debug,
    };
    pm.addPass(lwe::createAddDebugPort(addDebugPortOptions));

    // Convert LWE (and scheme-specific CKKS/BGV ops) to OpenFHE
    pm.addPass(lwe::createLWEToOpenfhe());
    pm.addPass(preprocessing::createPreprocessingToOpenfhe());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    auto configureCryptoContextOptions =
        openfhe::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.batchSize = options.batchSize;
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    configureCryptoContextOptions.firstModSize = options.firstModSize;
    configureCryptoContextOptions.insecure = options.insecure;
    configureCryptoContextOptions.mulDepth = options.mulDepth;
    configureCryptoContextOptions.ringDim = options.ringDim;
    configureCryptoContextOptions.scalingModSize = options.scalingModSize;
    configureCryptoContextOptions.scalingTechniqueFixedManual =
        options.scalingTechniqueFixedManual;
    pm.addPass(
        openfhe::createConfigureCryptoContext(configureCryptoContextOptions));

    pm.addPass(openfhe::createFastRotationPrecompute());
    // Vectorize any operations
    pm.addPass(createBooleanVectorizer());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(openfhe::createAllocToInPlace());

    pm.addPass(createRemoveUnusedPureCall());
    pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createSymbolDCEPass());
  };
}

BackendPipelineBuilder toLattigoPipelineBuilder() {
  return [=](OpPassManager& pm, const BackendOptions& options) {
    // Convert to (common trivial subset of) LWE
    // TODO (#1193): Replace `--bgv-to-lwe` with `--bgv-common-to-lwe`
    pm.addPass(bgv::createBGVToLWE());
    pm.addPass(ckks::createCKKSToLWE());

    if (!extConstOutputDir.empty()) {
      ExternalizeConstantsOptions extConstOptions;
      extConstOptions.outputDir = extConstOutputDir;
      extConstOptions.runtimeLoadDir = extConstRuntimeLoadDir;
      extConstOptions.thresholdElements = extConstThreshold;
      pm.addPass(createExternalizeConstants(extConstOptions));
    }

    // insert debug handler calls
    lwe::AddDebugPortOptions addDebugPortOptions{
        .entryFunction = options.entryFunction,
        .insertDebugAfterEveryOp = options.debug,
    };
    pm.addPass(lwe::createAddDebugPort(addDebugPortOptions));

    // Convert LWE (and scheme-specific BGV ops) to Lattigo
    pm.addPass(lwe::createLWEToLattigo());
    pm.addPass(preprocessing::createPreprocessingToLattigo());

    // Convert Alloc Ops to InPlace Ops
    pm.addPass(lattigo::createAllocToInPlace());

    // Simplify, in case the lowering revealed redundancy
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    auto configureCryptoContextOptions =
        lattigo::ConfigureCryptoContextOptions{};
    configureCryptoContextOptions.entryFunction = options.entryFunction;
    pm.addPass(
        lattigo::createConfigureCryptoContext(configureCryptoContextOptions));

    pm.addPass(createRemoveUnusedPureCall());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createSymbolDCEPass());

    // Bufferize without deallocation because golang has garbage collection.
    prepareForBufferize(pm);
    oneShotBufferize(pm, /*includeDeallocation=*/false);

    // Lower Linalg to loops
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  };
}

void linalgPreprocessingBuilder(OpPassManager& manager) {
  manager.addPass(createInlineActivations());
  manager.addPass(createActivationCanonicalizations());
  manager.addPass(createLinalgCanonicalizations());
  manager.addPass(createLinalgFuseLinearOpsPass());
  manager.addPass(createDropUnitDims());
  manager.addPass(createFoldConstantTensors());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSymbolDCEPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createLinalgCanonicalizations());
  manager.addPass(createReductionCanonicalizations());
  manager.addPass(createSoftmaxCanonicalizations());
}

void torchLinalgToCkksBuilder(OpPassManager& manager,
                              const MlirToRLWEPipelineOptions& options) {
  manager.addPass(debug::createDebugValidateNames());
  linalgPreprocessingBuilder(manager);
  MlirToRLWEPipelineOptions suboptions;

  suboptions.enableArithmetization = true;
  suboptions.minSlotCount = options.minSlotCount;
  suboptions.greedyBootstrapWaterline = options.greedyBootstrapWaterline;
  suboptions.useCompositeRelu = options.useCompositeRelu;
  suboptions.scalingModBits = options.scalingModBits;
  suboptions.firstModBits = options.firstModBits;
  suboptions.enableSplitPreprocessing = options.enableSplitPreprocessing;
  suboptions.experimentalDisableLoopUnroll =
      options.experimentalDisableLoopUnroll;
  suboptions.usePublicKey = options.usePublicKey;
  suboptions.encryptionTechniqueExtended = options.encryptionTechniqueExtended;
  suboptions.greedyModulusSwitchAfterMul = options.greedyModulusSwitchAfterMul;
  suboptions.greedyModulusSwitchBeforeFirstMul =
      options.greedyModulusSwitchBeforeFirstMul;
  suboptions.plaintextModulus = options.plaintextModulus;
  suboptions.noiseModel = options.noiseModel;
  suboptions.annotateNoiseBound = options.annotateNoiseBound;
  suboptions.bfvModBits = options.bfvModBits;
  suboptions.greedyLevelBudget = options.greedyLevelBudget;
  suboptions.plaintextExecutionResultFileName =
      options.plaintextExecutionResultFileName;
  suboptions.codegenStrategy = options.codegenStrategy;

  mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::ckksScheme)(manager,
                                                                suboptions);
}

}  // namespace mlir::heir
