#include <cstdlib>
#include <string>

#include "include/Conversion/BGVToOpenfhe/BGVToOpenfhe.h"
#include "include/Conversion/BGVToPolynomial/BGVToPolynomial.h"
#include "include/Conversion/CGGIToTfheRust/CGGIToTfheRust.h"
#include "include/Conversion/CombToCGGI/CombToCGGI.h"
#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "include/Conversion/PolynomialToStandard/PolynomialToStandard.h"
#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/CGGI/Transforms/Passes.h"
#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/LWE/Transforms/Passes.h"
#include "include/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "include/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "include/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "include/Dialect/Secret/Transforms/Passes.h"
#include "include/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "include/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "include/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "include/Transforms/Secretize/Passes.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToArith/TosaToArith.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToLinalg/TosaToLinalg.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToTensor/TosaToTensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"             // from @llvm-project
#include "mlir/include/mlir/InitAllPasses.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "include/Transforms/YosysOptimizer/YosysOptimizer.h"
#endif

using namespace mlir;
using namespace tosa;
using namespace heir;
using mlir::func::FuncOp;

void tosaToLinalg(OpPassManager &manager) {
  manager.addNestedPass<FuncOp>(createTosaToLinalgNamed());
  manager.addNestedPass<FuncOp>(createTosaToLinalg());
  manager.addNestedPass<FuncOp>(createTosaToArith(true, false));
  manager.addNestedPass<FuncOp>(createTosaToTensor());
  manager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  manager.addNestedPass<FuncOp>(createLinalgDetensorizePass());
  manager.addPass(createConvertTensorToLinalgPass());
  manager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
}

void oneShotBufferize(OpPassManager &manager) {
  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      bufferization::createOneShotBufferizePass(bufferizationOptions));
  manager.addPass(memref::createExpandReallocPass());
  manager.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(bufferization::createBufferDeallocationSimplificationPass());
  manager.addPass(bufferization::createLowerDeallocationsPass());
  manager.addPass(createCSEPass());
  manager.addPass(createCanonicalizerPass());
}

void tosaPipelineBuilder(OpPassManager &manager) {
  // TOSA to linalg
  tosaToLinalg(manager);
  // Bufferize
  manager.addNestedPass<FuncOp>(createLinalgBufferizePass());
  manager.addNestedPass<FuncOp>(tensor::createTensorBufferizePass());
  manager.addPass(arith::createArithBufferizePass());
  manager.addPass(func::createFuncBufferizePass());
  manager.addNestedPass<FuncOp>(bufferization::createFinalizingBufferizePass());
  // Affine
  manager.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(memref::createFoldMemRefAliasOpsPass());
  manager.addPass(createExpandCopyPass());
  manager.addPass(createExtractLoopBodyPass());
  manager.addPass(createUnrollAndForwardPass());
  // Cleanup
  manager.addPass(createMemrefGlobalReplacePass());
  arith::ArithIntNarrowingOptions options;
  options.bitwidthsSupported = {4, 8, 16};
  manager.addPass(arith::createArithIntNarrowing(options));
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

void polynomialToLLVMPipelineBuilder(OpPassManager &manager) {
  // Poly
  manager.addPass(polynomial::createPolynomialToStandard());
  manager.addPass(createCanonicalizerPass());

  // Linalg
  manager.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
  // Needed to lower affine.map and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

  // Bufferize
  oneShotBufferize(manager);

  // Linalg must be bufferized before it can be lowered
  // But lowering to loops also re-introduces affine.apply, so re-lower that
  manager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  manager.addPass(createLowerAffinePass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());

  // ToLLVM
  manager.addPass(arith::createArithExpandOpsPass());
  manager.addPass(createConvertSCFToCFPass());
  manager.addPass(createConvertControlFlowToLLVMPass());
  manager.addPass(createConvertIndexToLLVMPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createConvertFuncToLLVMPass());
  manager.addPass(createArithToLLVMConversionPass());
  manager.addPass(createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(createReconcileUnrealizedCastsPass());
  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

#ifndef HEIR_NO_YOSYS
struct TosaToBooleanTfheOptions
    : public PassPipelineOptions<TosaToBooleanTfheOptions> {
  PassOptions::Option<bool> abcFast{*this, "abc-fast",
                                    llvm::cl::desc("Run abc in fast mode."),
                                    llvm::cl::init(false)};

  PassOptions::Option<int> unrollFactor{
      *this, "unroll-factor",
      llvm::cl::desc("Unroll loops by a given factor before optimizing. A "
                     "value of zero (default) prevents unrolling."),
      llvm::cl::init(0)};

  PassOptions::Option<std::string> entryFunction{
      *this, "entry-function", llvm::cl::desc("Entry function to secretize"),
      llvm::cl::init("main")};
};

void tosaToBooleanTfhePipeline(const std::string &yosysFilesPath,
                               const std::string &abcPath) {
  PassPipelineRegistration<TosaToBooleanTfheOptions>(
      "tosa-to-boolean-tfhe", "Arithmetic modules to boolean tfhe-rs pipeline.",
      [yosysFilesPath, abcPath](OpPassManager &pm,
                                const TosaToBooleanTfheOptions &options) {
        // Secretize inputs
        pm.addPass(createSecretize(SecretizeOptions{options.entryFunction}));

        // TOSA to linalg
        tosaToLinalg(pm);

        // Bufferize
        oneShotBufferize(pm);

        // Affine
        pm.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());
        pm.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
        pm.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
        pm.addNestedPass<FuncOp>(memref::createExpandOpsPass());
        pm.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(createExpandCopyPass());

        // Cleanup
        pm.addPass(createMemrefGlobalReplacePass());
        arith::ArithIntNarrowingOptions arithOps;
        arithOps.bitwidthsSupported = {4, 8, 16};
        pm.addPass(arith::createArithIntNarrowing(arithOps));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createSCCPPass());
        pm.addPass(createCSEPass());
        pm.addPass(createSymbolDCEPass());

        // Wrap with secret.generic and then distribute-generic.
        pm.addPass(createWrapGeneric());
        auto distributeOpts = secret::SecretDistributeGenericOptions{};
        distributeOpts.opsToDistribute = {"affine.for", "affine.load",
                                          "affine.store", "memref.get_global"};
        pm.addPass(secret::createSecretDistributeGeneric(distributeOpts));
        pm.addPass(createCanonicalizerPass());

        // Booleanize and Yosys Optimize
        pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath,
                                        options.abcFast, options.unrollFactor));

        // Lower combinational circuit to CGGI
        pm.addPass(mlir::createCSEPass());
        pm.addPass(comb::createCombToCGGI());

        // CGGI to Tfhe-Rust exit dialect
        pm.addPass(createCGGIToTfheRust());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());

        // Cleanup loads and stores
        pm.addPass(createExpandCopyPass(
            ExpandCopyPassOptions{.disableAffineLoop = true}));
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(createForwardStoreToLoad());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createSCCPPass());
      });
}
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<bgv::BGVDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<poly_ext::PolyExtDialect>();
  registry.insert<polynomial::PolynomialDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();

  // Add expected MLIR dialects to the registry.
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<TosaDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registerAllDialects(registry);

  // Register MLIR core passes to build pipeline.
  registerAllPasses();

  // Custom passes in HEIR
  cggi::registerCGGIPasses();
  lwe::registerLWEPasses();
  secret::registerSecretPasses();
  registerSecretizePasses();
  registerFullLoopUnrollPasses();
  registerForwardStoreToLoadPasses();
  // Register yosys optimizer pipeline if configured.
#ifndef HEIR_NO_YOSYS
  const char *abcEnvPath = std::getenv("HEIR_ABC_BINARY");
  const char *yosysRunfilesEnvPath = std::getenv("HEIR_YOSYS_SCRIPTS_DIR");
  if (abcEnvPath == nullptr || yosysRunfilesEnvPath == nullptr) {
    llvm::errs() << "yosys optimizer deps not found, please set "
                    "HEIR_ABC_PATH and HEIR_YOSYS_LIBS; otherwise, set "
                    "HEIR_NO_YOSYS=1\n";
    return EXIT_FAILURE;
  }
  mlir::heir::registerYosysOptimizerPipeline(yosysRunfilesEnvPath, abcEnvPath);
  tosaToBooleanTfhePipeline(yosysRunfilesEnvPath, abcEnvPath);
#endif

  // Dialect conversion passes in HEIR
  bgv::registerBGVToPolynomialPasses();
  bgv::registerBGVToOpenfhePasses();
  comb::registerCombToCGGIPasses();
  polynomial::registerPolynomialToStandardPasses();
  registerCGGIToTfheRustPasses();

  PassPipelineRegistration<>(
      "heir-tosa-to-arith",
      "Run passes to lower TOSA models with stripped quant types to arithmetic",
      tosaPipelineBuilder);

  PassPipelineRegistration<>(
      "heir-polynomial-to-llvm",
      "Run passes to lower the polynomial dialect to LLVM",
      polynomialToLLVMPipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
