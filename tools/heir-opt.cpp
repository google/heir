#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"
#include "lib/Dialect/Arith/Conversions/ArithToCGGIQuart/ArithToCGGIQuart.h"
#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"
#include "lib/Dialect/BGV/Conversions/BGVToLWE/BGVToLWE.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToJaxite/CGGIToJaxite.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRustBool/CGGIToTfheRustBool.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/Transforms/Passes.h"
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/Transforms/Passes.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/Transforms/Passes.h"
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/Transforms/Passes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/Transforms/Passes.h"
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/Transforms/Passes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"
#include "lib/Dialect/Secret/Conversions/SecretToCGGI/SecretToCGGI.h"
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/Transforms/BufferizableOpInterfaceImpl.h"
#include "lib/Dialect/Secret/Transforms/Passes.h"
#include "lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.h"
#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/Transforms/Passes.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Pipelines/ArithmeticPipelineRegistration.h"
#include "lib/Pipelines/BooleanPipelineRegistration.h"
#include "lib/Pipelines/PipelineRegistration.h"
#include "lib/Transforms/AddClientInterface/AddClientInterface.h"
#include "lib/Transforms/AnnotateModule/AnnotateModule.h"
#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h"
#include "lib/Transforms/ApplyFolders/ApplyFolders.h"
#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h"
#include "lib/Transforms/ConvertSecretExtractToStaticExtract/ConvertSecretExtractToStaticExtract.h"
#include "lib/Transforms/ConvertSecretForToStaticFor/ConvertSecretForToStaticFor.h"
#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h"
#include "lib/Transforms/ConvertSecretWhileToStaticFor/ConvertSecretWhileToStaticFor.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"
#include "lib/Transforms/DropUnitDims/DropUnitDims.h"
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"
#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h"
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h"
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"
#include "lib/Transforms/LowerUnpack/LowerUnpack.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/SelectRewrite/SelectRewrite.h"
#include "lib/Transforms/SetPlaintextParams/SetPlaintextParams.h"
#include "lib/Transforms/ShapeInference/ShapeInference.h"
#include "lib/Transforms/StraightLineVectorizer/StraightLineVectorizer.h"
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h"
#include "lib/Transforms/TensorToScalars/TensorToScalars.h"
#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h"
#include "lib/Transforms/ValidateNoise/ValidateNoise.h"
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/UBToLLVM/UBToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#endif

// This comment includes internal conversions
// This comment includes internal dialects
// This comment includes internal pipelines

using namespace mlir;
using namespace tosa;
using namespace heir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // This comment inserts internal dialects

  registry.insert<mod_arith::ModArithDialect>();
  registry.insert<bgv::BGVDialect>();
  registry.insert<ckks::CKKSDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<jaxite::JaxiteDialect>();
  registry.insert<jaxiteword::JaxiteWordDialect>();
  registry.insert<lattigo::LattigoDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<mgmt::MgmtDialect>();
  registry.insert<random::RandomDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<rns::RNSDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();

  // Add expected MLIR dialects to the registry.
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<::mlir::emitc::EmitCDialect>();
  registry.insert<::mlir::linalg::LinalgDialect>();
  registry.insert<TosaDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<::mlir::heir::polynomial::PolynomialDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();

  // Uncomment if you want everything bound to CLI flags.
  // registerAllDialects(registry);
  // registerAllExtensions(registry);
  // registerAllPasses();

  // Upstream passes used by HEIR
  // Converting to LLVM
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  func::registerAllExtensions(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);

  // Misc
  registerTransformsPasses();      // canonicalize, cse, etc.
  affine::registerAffinePasses();  // loop unrolling
  registerLinalgPasses();          // linalg to loops

  // These are only needed by two tests that build a pass pipeline
  // from the CLI. Those tests can probably eventually be removed.
  //   - `tests/memref_global.mlir`
  //   - `tests/memref_global_raw.mlir`
  registerPass([]() -> std::unique_ptr<Pass> {
    return createArithToLLVMConversionPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createConvertControlFlowToLLVMPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertFuncToLLVMPass(); });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createSCFToControlFlowPass(); });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createFinalizeMemRefToLLVMConversionPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createLowerAffinePass(); });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createReconcileUnrealizedCastsPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertToLLVMPass(); });

  // Bufferization and external models
  bufferization::registerBufferizationPasses();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);

  // Custom passes in HEIR
  cggi::registerCGGIPasses();
  lattigo::registerLattigoPasses();
  lwe::registerLWEPasses();
  mgmt::registerMgmtPasses();
  openfhe::registerOpenfhePasses();
  ::mlir::heir::polynomial::registerPolynomialPasses();
  secret::registerSecretPasses();
  tensor_ext::registerTensorExtPasses();
  registerAddClientInterfacePass();
  registerElementwiseToAffinePasses();
  registerSecretizePasses();
  registerSecretInsertMgmtPasses();
  registerFullLoopUnrollPasses();
  registerConvertIfToSelectPasses();
  registerSelectRewritePasses();
  registerConvertSecretForToStaticForPasses();
  registerConvertSecretWhileToStaticForPasses();
  registerConvertSecretExtractToStaticExtractPasses();
  registerConvertSecretInsertToStaticInsertPasses();
  registerConvertToCiphertextSemanticsPasses();
  registerDropUnitDims();
  registerAnnotateModulePasses();
  registerAnnotateSecretnessPasses();
  registerApplyFoldersPasses();
  registerForwardInsertToExtractPasses();
  registerForwardStoreToLoadPasses();
  registerGenerateParamPasses();
  registerOperationBalancerPasses();
  registerPopulateScalePasses();
  registerStraightLineVectorizerPasses();
  registerSetPlaintextParams();
  registerUnusedMemRefPasses();
  registerValidateNoisePasses();
  registerOptimizeRelinearizationPasses();
  registerPolynomialApproximationPasses();
  registerPropagateAnnotationPasses();
  registerLayoutPropagationPasses();
  registerLayoutOptimizationPasses();
  registerLinalgCanonicalizationsPasses();
  registerLowerPolynomialEvalPasses();
  registerLowerUnpackPasses();
  registerTensorToScalarsPasses();
  registerTensorLinalgToAffineLoops();
  registerShapeInferencePasses();
  // Register yosys optimizer pipeline if configured.
#ifndef HEIR_NO_YOSYS
#ifndef HEIR_ABC_BINARY
  llvm::errs() << "HEIR_ABC_BINARY #define not properly set";
  return EXIT_FAILURE;
#endif
#ifndef HEIR_YOSYS_SCRIPTS_DIR
  llvm::errs() << "HEIR_YOSYS_SCRIPTS_DIR #define not properly set";
  return EXIT_FAILURE;
#endif
  const char *abcEnvPath = HEIR_ABC_BINARY;
  const char *yosysRunfilesEnvPath = HEIR_YOSYS_SCRIPTS_DIR;
  // When running in a lit test, these #defines must be overridden
  // by environment variables set in tests/lit.cfg.py
  char *overriddenAbcEnvPath = std::getenv("HEIR_ABC_BINARY");
  char *overriddenYosysRunfilesEnvPath = std::getenv("HEIR_YOSYS_SCRIPTS_DIR");
  if (overriddenAbcEnvPath != nullptr) abcEnvPath = overriddenAbcEnvPath;
  if (overriddenYosysRunfilesEnvPath != nullptr)
    yosysRunfilesEnvPath = overriddenYosysRunfilesEnvPath;
  mlir::heir::registerYosysOptimizerPipeline(yosysRunfilesEnvPath, abcEnvPath);

  PassPipelineRegistration<mlir::heir::MLIRToCGGIPipelineOptions>(
      "mlir-to-cggi",
      "Convert a func using standard MLIR dialects to FHE using "
      "CGGI.",
      mlirToCGGIPipelineBuilder(yosysRunfilesEnvPath, abcEnvPath));
#else
  PassPipelineRegistration<mlir::heir::MLIRToCGGIPipelineOptions>(
      "mlir-to-cggi",
      "Convert a func using standard MLIR dialects to FHE using "
      "CGGI.",
      mlirToCGGIPipelineBuilder());
#endif

  // Dialect conversion passes in HEIR
  mod_arith::registerModArithToArithPasses();
  mlir::heir::arith::registerArithToModArithPasses();
  mlir::heir::arith::registerArithToCGGIPasses();
  mlir::heir::arith::registerArithToCGGIQuartPasses();
  mod_arith::registerConvertToMacPass();
  bgv::registerBGVToLWEPasses();
  ckks::registerCKKSToLWEPasses();
  registerSecretToCGGIPasses();
  lwe::registerLWEToLattigoPasses();
  lwe::registerLWEToOpenfhePasses();
  lwe::registerLWEToPolynomialPasses();
  ::mlir::heir::linalg::registerLinalgToTensorExtPasses();
  ::mlir::heir::polynomial::registerPolynomialToModArithPasses();
  tensor_ext::registerTensorExtToTensorPasses();
  registerCGGIToJaxitePasses();
  registerCGGIToTfheRustPasses();
  registerCGGIToTfheRustBoolPasses();
  // This comement registers internal passes
  registerSecretToBGVPasses();
  registerSecretToCKKSPasses();
  registerSecretToModArithPasses();
  ::mlir::heir::tosa::registerTosaToSecretArithPasses();

  // Interfaces in HEIR
  secret::registerBufferizableOpInterfaceExternalModels(registry);
  rns::registerExternalRNSTypeInterfaces(registry);
  registerOperandAndResultAttrInterface(registry);

  PassPipelineRegistration<TosaToArithOptions>(
      "heir-tosa-to-arith",
      "Run passes to lower TOSA models with stripped "
      "quant types to arithmetic",
      [](OpPassManager &pm, const TosaToArithOptions &options) {
        ::mlir::heir::tosaPipelineBuilder(pm, options.unroll);
      });

  PassPipelineRegistration<>(
      "heir-polynomial-to-llvm",
      "Run passes to lower the polynomial dialect to LLVM",
      ::mlir::heir::polynomialToLLVMPipelineBuilder);

  PassPipelineRegistration<>("heir-basic-mlir-to-llvm",
                             "Lower basic MLIR to LLVM",
                             ::mlir::heir::basicMLIRToLLVMPipelineBuilder);

  PassPipelineRegistration<SimdVectorizerOptions>(
      "heir-simd-vectorizer",
      "Run scheme-agnostic passes to convert FHE programs that operate on "
      "scalar types to equivalent programs that operate on vectors and use "
      "tensor_ext.rotate",
      [](OpPassManager &pm, const SimdVectorizerOptions &options) {
        ::mlir::heir::heirSIMDVectorizerPipelineBuilder(
            pm, options.experimentalDisableLoopUnroll);
      });

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-secret-arithmetic",
      "Convert a func using standard MLIR dialects to secret dialect with "
      "arithmetic ops",
      [](OpPassManager &pm,
         const mlir::heir::MlirToRLWEPipelineOptions &options) {
        mlirToSecretArithmeticPipelineBuilder(pm, options);
      });

  PassPipelineRegistration<mlir::heir::PlaintextBackendOptions>(
      "mlir-to-plaintext-backend",
      "Convert a func using standard MLIR dialects to plaintext backend "
      "executed in secret arithmetic",
      [](OpPassManager &pm,
         const mlir::heir::PlaintextBackendOptions &options) {
        mlirToPlaintextPipelineBuilder(pm, options);
      });

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-bgv",
      "Convert a func using standard MLIR dialects to FHE using "
      "BGV.",
      mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::bgvScheme));

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-bfv",
      "Convert a func using standard MLIR dialects to FHE using "
      "BFV.",
      mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::bfvScheme));

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-ckks",
      "Convert a func using standard MLIR dialects to FHE using "
      "CKKS.",
      mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::ckksScheme));

  PassPipelineRegistration<mlir::heir::BackendOptions>(
      "scheme-to-openfhe",
      "Convert code expressed at FHE scheme level to OpenFHE C++ code.",
      toOpenFhePipelineBuilder());

  PassPipelineRegistration<mlir::heir::BackendOptions>(
      "scheme-to-lattigo",
      "Convert code expressed at FHE scheme level to Lattigo Go code.",
      toLattigoPipelineBuilder());

  // TODO(#1645): Add backend options for tfhe-rs, fpt, jaxite.
  PassPipelineRegistration<>(
      "scheme-to-tfhe-rs",
      "Convert code expressed at FHE scheme level to tfhe-rs code.",
      toTfheRsPipelineBuilder());

  PassPipelineRegistration<>(
      "scheme-to-fpt",
      "Convert code expressed at FHE scheme level to Lattigo Go code.",
      toFptPipelineBuilder());

  PassPipelineRegistration<mlir::heir::CGGIBackendOptions>(
      "scheme-to-jaxite",
      "Convert code expressed at FHE scheme level to jaxite code.",
      toJaxitePipelineBuilder());

  // Register internal pipeline

  PassPipelineRegistration<>(
      "convert-to-data-oblivious",
      "Transforms a native program to data-oblivious program",
      convertToDataObliviousPipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
