#include "include/Conversion/BGVToPoly/BGVToPoly.h"
#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "include/Conversion/PolyToStandard/PolyToStandard.h"
#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/Poly/IR/PolyDialect.h"
#include "include/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/Transforms/Passes.h"
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/TosaToLinalg/TosaToLinalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"             // from @llvm-project
#include "mlir/include/mlir/InitAllPasses.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

using namespace mlir;
using namespace tosa;
using namespace heir;
using mlir::func::FuncOp;

void tosaPipelineBuilder(OpPassManager &manager) {
  // TOSA to linalg
  manager.addNestedPass<FuncOp>(createTosaToLinalgNamed());
  manager.addNestedPass<FuncOp>(createTosaToLinalg());
  manager.addNestedPass<FuncOp>(createTosaToArith(true, false));
  manager.addNestedPass<FuncOp>(createTosaToTensor());
  manager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  manager.addNestedPass<FuncOp>(createLinalgDetensorizePass());
  manager.addPass(createConvertTensorToLinalgPass());
  manager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
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
  manager.addPass(createUnrollAndForwardStoresPass());
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

void polyToLLVMPipelineBuilder(OpPassManager &manager) {
  // Poly
  manager.addPass(poly::createPolyToStandard());
  manager.addPass(createCanonicalizerPass());

  // Linalg
  manager.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
  // Needed to lower affine.map and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

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

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<bgv::BGVDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<poly::PolyDialect>();
  registry.insert<poly_ext::PolyExtDialect>();
  registry.insert<secret::SecretDialect>();

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
  secret::registerSecretPasses();

  // Custom passes in HEIR
  poly::registerPolyToStandardPasses();
  bgv::registerBGVToPolyPasses();

  PassPipelineRegistration<>(
      "heir-tosa-to-arith",
      "Run passes to lower TOSA models with stripped quant types to arithmetic",
      tosaPipelineBuilder);

  PassPipelineRegistration<>(
      "heir-polynomial-to-llvm",
      "Run passes to lower the polynomial dialect to LLVM",
      polyToLLVMPipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
