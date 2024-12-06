#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include "lib/Dialect/BGV/Conversions/BGVToLWE/BGVToLWE.h"
#include "lib/Dialect/BGV/Conversions/BGVToOpenfhe/BGVToOpenfhe.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToJaxite/CGGIToJaxite.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h"
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRustBool/CGGIToTfheRustBool.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/Transforms/Passes.h"
#include "lib/Dialect/CKKS/Conversions/CKKSToOpenfhe/CKKSToOpenfhe.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/Transforms/Passes.h"
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h"
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/Transforms/Passes.h"
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/Transforms/Passes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"
#include "lib/Dialect/Secret/Conversions/SecretToCGGI/SecretToCGGI.h"
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/Transforms/BufferizableOpInterfaceImpl.h"
#include "lib/Dialect/Secret/Transforms/Passes.h"
#include "lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/Transforms/Passes.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Pipelines/ArithmeticPipelineRegistration.h"
#include "lib/Pipelines/PipelineRegistration.h"
#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h"
#include "lib/Transforms/ApplyFolders/ApplyFolders.h"
#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h"
#include "lib/Transforms/ConvertSecretExtractToStaticExtract/ConvertSecretExtractToStaticExtract.h"
#include "lib/Transforms/ConvertSecretForToStaticFor/ConvertSecretForToStaticFor.h"
#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h"
#include "lib/Transforms/ConvertSecretWhileToStaticFor/ConvertSecretWhileToStaticFor.h"
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"
#include "lib/Transforms/ForwardInsertToExtract/ForwardInsertToExtract.h"
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/StraightLineVectorizer/StraightLineVectorizer.h"
#include "lib/Transforms/TensorToScalars/TensorToScalars.h"
#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h"
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/UBToLLVM/UBToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
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
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "lib/Pipelines/BooleanPipelineRegistration.h"
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#endif

#include "lib/Dialect/Optalysys/IR/OptalysysDialect.h"

using namespace mlir;
using namespace tosa;
using namespace heir;
using mlir::func::FuncOp;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<optalysys::OptalysysDialect>();

  registry.insert<mod_arith::ModArithDialect>();
  registry.insert<bgv::BGVDialect>();
  registry.insert<ckks::CKKSDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<jaxite::JaxiteDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<random::RandomDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<rns::RNSDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();

  // Add expected MLIR dialects to the registry.
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<::mlir::linalg::LinalgDialect>();
  registry.insert<TosaDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
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
  arith::registerConvertArithToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  func::registerAllExtensions(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);

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
      []() -> std::unique_ptr<Pass> { return createConvertSCFToCFPass(); });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createFinalizeMemRefToLLVMConversionPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createLowerAffinePass(); });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createReconcileUnrealizedCastsPass();
  });

  // Bufferization and external models
  bufferization::registerBufferizationPasses();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);

  // Custom passes in HEIR
  cggi::registerCGGIPasses();
  lwe::registerLWEPasses();
  ::mlir::heir::polynomial::registerPolynomialPasses();
  secret::registerSecretPasses();
  tensor_ext::registerTensorExtPasses();
  openfhe::registerOpenfhePasses();
  registerElementwiseToAffinePasses();
  registerSecretizePasses();
  registerFullLoopUnrollPasses();
  registerConvertIfToSelectPasses();
  registerConvertSecretForToStaticForPasses();
  registerConvertSecretWhileToStaticForPasses();
  registerConvertSecretExtractToStaticExtractPasses();
  registerConvertSecretInsertToStaticInsertPasses();
  registerAnnotateSecretnessPasses();
  registerApplyFoldersPasses();
  registerForwardInsertToExtractPasses();
  registerForwardStoreToLoadPasses();
  registerOperationBalancerPasses();
  registerStraightLineVectorizerPasses();
  registerUnusedMemRefPasses();
  registerOptimizeRelinearizationPasses();
  registerLinalgCanonicalizationsPasses();
  registerTensorToScalarsPasses();
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
  registerTosaToBooleanTfhePipeline(yosysRunfilesEnvPath, abcEnvPath);
  registerTosaToBooleanFpgaTfhePipeline(yosysRunfilesEnvPath, abcEnvPath);
  registerTosaToJaxitePipeline(yosysRunfilesEnvPath, abcEnvPath);
#endif

  // Dialect conversion passes in HEIR
  mod_arith::registerModArithToArithPasses();
  bgv::registerBGVToLWEPasses();
  bgv::registerBGVToOpenfhePasses();
  ckks::registerCKKSToOpenfhePasses();
  registerSecretToCGGIPasses();
  lwe::registerLWEToPolynomialPasses();
  ::mlir::heir::linalg::registerLinalgToTensorExtPasses();
  ::mlir::heir::polynomial::registerPolynomialToModArithPasses();
  registerCGGIToJaxitePasses();
  registerCGGIToTfheRustPasses();
  registerCGGIToTfheRustBoolPasses();
  registerSecretToBGVPasses();
  registerSecretToCKKSPasses();
  mlir::heir::tosa::registerTosaToSecretArithPasses();

  // Interfaces in HEIR
  secret::registerBufferizableOpInterfaceExternalModels(registry);
  rns::registerExternalRNSTypeInterfaces(registry);

  PassPipelineRegistration<>("heir-tosa-to-arith",
                             "Run passes to lower TOSA models with stripped "
                             "quant types to arithmetic",
                             ::mlir::heir::tosaPipelineBuilder);

  PassPipelineRegistration<>(
      "heir-polynomial-to-llvm",
      "Run passes to lower the polynomial dialect to LLVM",
      ::mlir::heir::polynomialToLLVMPipelineBuilder);

  PassPipelineRegistration<>("heir-basic-mlir-to-llvm",
                             "Lower basic MLIR to LLVM",
                             ::mlir::heir::basicMLIRToLLVMPipelineBuilder);

  PassPipelineRegistration<>(
      "heir-simd-vectorizer",
      "Run scheme-agnostic passes to convert FHE programs that operate on "
      "scalar types to equivalent programs that operate on vectors and use "
      "tensor_ext.rotate",
      mlir::heir::heirSIMDVectorizerPipelineBuilder);

  PassPipelineRegistration<>(
      "mlir-to-secret-arithmetic",
      "Convert a func using standard MLIR dialects to secret dialect with "
      "arithmetic ops",
      mlirToSecretArithmeticPipelineBuilder);

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-bgv",
      "Convert a func using standard MLIR dialects to FHE using "
      "BGV.",
      mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::bgvScheme));

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-openfhe-bgv",
      "Convert a func using standard MLIR dialects to FHE using BGV and "
      "export "
      "to OpenFHE C++ code.",
      mlirToOpenFheRLWEPipelineBuilder(mlir::heir::RLWEScheme::bgvScheme));

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-ckks",
      "Convert a func using standard MLIR dialects to FHE using "
      "CKKS.",
      mlirToRLWEPipelineBuilder(mlir::heir::RLWEScheme::ckksScheme));

  PassPipelineRegistration<mlir::heir::MlirToRLWEPipelineOptions>(
      "mlir-to-openfhe-ckks",
      "Convert a func using standard MLIR dialects to FHE using CKKS and "
      "export "
      "to OpenFHE C++ code.",
      mlirToOpenFheRLWEPipelineBuilder(mlir::heir::RLWEScheme::ckksScheme));

  PassPipelineRegistration<>(
      "convert-to-data-oblivious",
      "Transforms a native program to data-oblivious program",
      convertToDataObliviousPipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
