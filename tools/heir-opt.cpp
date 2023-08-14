#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "include/Conversion/PolyToStandard/PolyToStandard.h"
#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/EncryptedArith/IR/EncryptedArithDialect.h"
#include "include/Dialect/Poly/IR/PolyDialect.h"
#include "mlir/include/mlir/Conversion/TosaToLinalg/TosaToLinalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"             // from @llvm-project
#include "mlir/include/mlir/InitAllPasses.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

void tosaPipelineBuilder(mlir::OpPassManager &manager) {
  // TOSA to linalg
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaToLinalgNamed());
  manager.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaToArith(true, false));
  manager.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToTensor());
  manager.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgDetensorizePass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());
  manager.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  // Bufferize
  manager.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgBufferizePass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::tensor::createTensorBufferizePass());
  manager.addPass(mlir::arith::createArithBufferizePass());
  manager.addPass(mlir::func::createFuncBufferizePass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  // Affine
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToAffineLoopsPass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::memref::createExpandStridedMetadataPass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::memref::createExpandOpsPass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createSimplifyAffineStructuresPass());
  manager.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  manager.addPass(mlir::heir::createExpandCopyPass());
  manager.addPass(mlir::heir::createExtractLoopBodyPass());
  manager.addPass(mlir::heir::createUnrollAndForwardStoresPass());
  // Cleanup
  manager.addPass(mlir::heir::createMemrefGlobalReplacePass());
  mlir::arith::ArithIntNarrowingOptions options;
  options.bitwidthsSupported = {4, 8, 16};
  manager.addPass(mlir::arith::createArithIntNarrowing(options));
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::heir::poly::PolyDialect>();
  registry.insert<mlir::heir::EncryptedArithDialect>();
  registry.insert<mlir::heir::bgv::BGVDialect>();

  // Add expected MLIR dialects to the registry.
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  mlir::registerAllDialects(registry);
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::tosa::TosaDialect>();

  // Register MLIR core passes to build pipeline.
  mlir::registerAllPasses();

  // Custom passes in HEIR
  mlir::heir::poly::registerPolyToStandardPasses();

  mlir::PassPipelineRegistration<>(
      "heir-tosa-to-arith",
      "Run passes to lower TOSA models with stripped quant types to arithmetic",
      tosaPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
