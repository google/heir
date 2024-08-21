#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "lib/Conversion/BGVToLWE/BGVToLWE.h"
#include "lib/Conversion/BGVToOpenfhe/BGVToOpenfhe.h"
#include "lib/Conversion/CGGIToJaxite/CGGIToJaxite.h"
#include "lib/Conversion/CGGIToTfheRust/CGGIToTfheRust.h"
#include "lib/Conversion/CGGIToTfheRustBool/CGGIToTfheRustBool.h"
#include "lib/Conversion/CombToCGGI/CombToCGGI.h"
#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h"
#include "lib/Conversion/MemrefToArith/MemrefToArith.h"
#include "lib/Conversion/ModArithToArith/ModArithToArith.h"
#include "lib/Conversion/PolynomialToStandard/PolynomialToStandard.h"
#include "lib/Conversion/SecretToBGV/SecretToBGV.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/Transforms/Passes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/Transforms/AddClientInterface.h"
#include "lib/Dialect/LWE/Transforms/Passes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/Passes.h"
#include "lib/Dialect/Polynomial/Transforms/NTTRewrites.h"
#include "lib/Dialect/Polynomial/Transforms/Passes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/Transforms/BufferizableOpInterfaceImpl.h"
#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "lib/Dialect/Secret/Transforms/Passes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"
#include "lib/Dialect/TensorExt/Transforms/InsertRotate.h"
#include "lib/Dialect/TensorExt/Transforms/Passes.h"
#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Transforms/ApplyFolders/ApplyFolders.h"
#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h"
#include "lib/Transforms/ConvertSecretExtractToStaticExtract/ConvertSecretExtractToStaticExtract.h"
#include "lib/Transforms/ConvertSecretForToStaticFor/ConvertSecretForToStaticFor.h"
#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h"
#include "lib/Transforms/ConvertSecretWhileToStaticFor/ConvertSecretWhileToStaticFor.h"
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"
#include "lib/Transforms/OperationBalancer/OperationBalancer.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "lib/Transforms/StraightLineVectorizer/StraightLineVectorizer.h"
#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"  // from @llvm-project
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
#include "mlir/include/mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToArith/TosaToArith.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToLinalg/TosaToLinalg.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToTensor/TosaToTensor.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/UBToLLVM/UBToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"
#endif

using namespace mlir;
using namespace tosa;
using namespace heir;
using mlir::func::FuncOp;

static std::vector<std::string> opsToDistribute = {
    "affine.for",   "affine.load",       "memref.load",    "memref.store",
    "affine.store", "memref.get_global", "memref.dealloc", "memref.alloc"};

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
  oneShotBufferize(manager);
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
  manager.addPass(createElementwiseToAffine());
  manager.addPass(::mlir::heir::polynomial::createPolynomialToStandard());
  manager.addPass(createCanonicalizerPass());

  // Linalg
  manager.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
  // Needed to lower affine.map and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(createLowerAffinePass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

  // Bufferize
  oneShotBufferize(manager);

  // Linalg must be bufferized before it can be lowered
  // But lowering to loops also re-introduces affine.apply, so re-lower that
  manager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createBufferizationToMemRefPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());

  // ToLLVM
  manager.addPass(arith::createArithExpandOpsPass());
  manager.addPass(createConvertSCFToCFPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  manager.addPass(createConvertToLLVMPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

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
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
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
        pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass(true));
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
        auto distributeOpts = secret::SecretDistributeGenericOptions{
            .opsToDistribute = opsToDistribute};
        pm.addPass(secret::createSecretDistributeGeneric(distributeOpts));
        pm.addPass(createCanonicalizerPass());

        // Booleanize and Yosys Optimize
        pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath,
                                        options.abcFast, options.unrollFactor));

        // Lower combinational circuit to CGGI
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createSCCPPass());

        pm.addPass(mlir::createCSEPass());
        pm.addPass(secret::createSecretDistributeGeneric());
        pm.addPass(comb::createCombToCGGI());

        // CGGI to Tfhe-Rust exit dialect
        pm.addPass(createCGGIToTfheRust());
        // CSE must be run before canonicalizer, so that redundant ops are
        // cleared before the canonicalizer hoists TfheRust ops.
        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

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

struct TosaToBooleanFpgaTfheOptions
    : public PassPipelineOptions<TosaToBooleanFpgaTfheOptions> {
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

void tosaToBooleanFpgaTfhePipeline(const std::string &yosysFilesPath,
                                   const std::string &abcPath) {
  PassPipelineRegistration<TosaToBooleanFpgaTfheOptions>(
      "tosa-to-boolean-fpga-tfhe",
      "Arithmetic modules to boolean tfhe-rs for FPGA backend pipeline.",
      [yosysFilesPath, abcPath](OpPassManager &pm,
                                const TosaToBooleanFpgaTfheOptions &options) {
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
        pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass(true));
        pm.addNestedPass<FuncOp>(affine::createLoopFusionPass(
            0, 0, true, affine::FusionMode::Greedy));
        pm.addPass(affine::createAffineScalarReplacementPass());
        pm.addPass(createForwardStoreToLoad());

        // Cleanup
        pm.addPass(createMemrefGlobalReplacePass());
        arith::ArithIntNarrowingOptions arithOps;
        arithOps.bitwidthsSupported = {4, 8, 16};
        pm.addPass(arith::createArithIntNarrowing(arithOps));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createSCCPPass());
        pm.addPass(createCSEPass());
        pm.addPass(createSymbolDCEPass());

        pm.addPass(createWrapGeneric());
        auto distributeOpts = secret::SecretDistributeGenericOptions{
            .opsToDistribute = opsToDistribute};
        pm.addPass(secret::createSecretDistributeGeneric(distributeOpts));
        pm.addPass(createCanonicalizerPass());

        // Booleanize and Yosys Optimize
        pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath,
                                        options.abcFast, options.unrollFactor,
                                        Mode::Boolean));

        // Lower combinational circuit to CGGI
        pm.addPass(createForwardStoreToLoad());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(secret::createSecretDistributeGeneric());
        pm.addPass(comb::createCombToCGGI());
        // Cleanup CombToCGGI
        pm.addPass(createExpandCopyPass(
            ExpandCopyPassOptions{.disableAffineLoop = true}));
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(createForwardStoreToLoad());
        pm.addPass(createRemoveUnusedMemRef());

        pm.addPass(createStraightLineVectorizer(
            StraightLineVectorizerOptions{.dialect = "cggi"}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createSCCPPass());

        // CGGI to Tfhe-Rust exit dialect
        pm.addPass(createCGGIToTfheRustBool());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createSCCPPass());
      });
}
#endif

struct MlirToBgvPipelineOptions
    : public PassPipelineOptions<MlirToBgvPipelineOptions> {
  PassOptions::Option<std::string> entryFunction{
      *this, "entry-function", llvm::cl::desc("Entry function to secretize"),
      llvm::cl::init("main")};
  PassOptions::Option<int> ciphertextDegree{
      *this, "ciphertext-degree",
      llvm::cl::desc("The degree of the polynomials to use for ciphertexts; "
                     "equivalently, the number of messages that can be packed "
                     "into a single ciphertext."),
      llvm::cl::init(1024)};
};

void mlirToBgvPipelineBuilder(OpPassManager &pm,
                              const MlirToBgvPipelineOptions &options) {
  // Secretize inputs
  pm.addPass(createSecretize(SecretizeOptions{options.entryFunction}));
  pm.addPass(createWrapGeneric());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Vectorize and optimize rotations
  heirSIMDVectorizerPipelineBuilder(pm);

  // Prepare to lower to BGV
  pm.addPass(secret::createSecretDistributeGeneric());
  pm.addPass(createCanonicalizerPass());

  // Lower to BGV
  auto secretToBgvOpts = SecretToBGVOptions{};
  secretToBgvOpts.polyModDegree = options.ciphertextDegree;
  pm.addPass(createSecretToBGV(secretToBgvOpts));
}

void mlirToOpenFheBgvPipelineBuilder(OpPassManager &pm,
                                     const MlirToBgvPipelineOptions &options) {
  // lower to BGV
  mlirToBgvPipelineBuilder(pm, options);

  // Add client interface
  auto addClientInterfaceOptions = lwe::AddClientInterfaceOptions{};
  // OpenFHE's pke API, which this pipeline generates, is always public-key
  addClientInterfaceOptions.usePublicKey = true;
  addClientInterfaceOptions.oneValuePerHelperFn = true;
  pm.addPass(lwe::createAddClientInterface(addClientInterfaceOptions));

  // Lower to openfhe
  pm.addPass(bgv::createBGVToOpenfhe());
  pm.addPass(createCanonicalizerPass());
  auto configureCryptoContextOptions = openfhe::ConfigureCryptoContextOptions{};
  configureCryptoContextOptions.entryFunction = options.entryFunction;
  pm.addPass(
      openfhe::createConfigureCryptoContext(configureCryptoContextOptions));
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

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
  registry.insert<TosaDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<::mlir::polynomial::PolynomialDialect>();
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
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
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
  registerApplyFoldersPasses();
  registerForwardStoreToLoadPasses();
  registerOperationBalancerPasses();
  registerStraightLineVectorizerPasses();
  registerUnusedMemRefPasses();
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
  tosaToBooleanTfhePipeline(yosysRunfilesEnvPath, abcEnvPath);
  tosaToBooleanFpgaTfhePipeline(yosysRunfilesEnvPath, abcEnvPath);
#endif

  // Dialect conversion passes in HEIR
  mod_arith::registerModArithToArithPasses();
  bgv::registerBGVToLWEPasses();
  bgv::registerBGVToOpenfhePasses();
  comb::registerCombToCGGIPasses();
  lwe::registerLWEToPolynomialPasses();
  ::mlir::heir::polynomial::registerPolynomialToStandardPasses();
  registerCGGIToJaxitePasses();
  registerCGGIToTfheRustPasses();
  registerCGGIToTfheRustBoolPasses();
  registerSecretToBGVPasses();

  // Interfaces in HEIR
  secret::registerBufferizableOpInterfaceExternalModels(registry);
  rns::registerExternalRNSTypeInterfaces(registry);

  PassPipelineRegistration<>("heir-tosa-to-arith",
                             "Run passes to lower TOSA models with stripped "
                             "quant types to arithmetic",
                             tosaPipelineBuilder);

  PassPipelineRegistration<>(
      "heir-polynomial-to-llvm",
      "Run passes to lower the polynomial dialect to LLVM",
      polynomialToLLVMPipelineBuilder);

  PassPipelineRegistration<>(
      "heir-simd-vectorizer",
      "Run scheme-agnostic passes to convert FHE programs that operate on "
      "scalar types to equivalent programs that operate on vectors and use "
      "tensor_ext.rotate",
      heirSIMDVectorizerPipelineBuilder);

  PassPipelineRegistration<MlirToBgvPipelineOptions>(
      "mlir-to-bgv",
      "Convert a func using standard MLIR dialects to FHE using "
      "BGV.",
      mlirToBgvPipelineBuilder);

  PassPipelineRegistration<MlirToBgvPipelineOptions>(
      "mlir-to-openfhe-bgv",
      "Convert a func using standard MLIR dialects to FHE using BGV and "
      "export "
      "to OpenFHE C++ code.",
      mlirToOpenFheBgvPipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
