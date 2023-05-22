#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "include/Dialect/EncryptedArith/IR/EncryptedArithDialect.h"
#include "include/Dialect/HEIR/IR/HEIRDialect.h"
#include "mlir/include/mlir/Conversion/TosaToLinalg/TosaToLinalg.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Passes.h" // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h" // from @llvm-project
#include "mlir/include/mlir/InitAllPasses.h" // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h" // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h" // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h" // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h" // from @llvm-project

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
  mlir::ConvertMathToFuncsOptions mathToFuncsOptions{};
  mathToFuncsOptions.convertCtlz = true;
  manager.addPass(mlir::createConvertMathToFuncs(mathToFuncsOptions));
  manager.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  manager.addPass(mlir::heir::createExpandCopyPass());
  // Loop unroll
  // FIXME: Develop a custom pass that can greedily unroll affine loops.
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollPass(-1, false, true));
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollPass(-1, false, true));
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollPass(-1, false, true));
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollPass(-1, false, true));
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollPass(-1, false, true));
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollPass(-1, false, true));
  // Cleanup
  manager.addPass(mlir::heir::createMemrefGlobalReplacePass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createAffineScalarReplacementPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::heir::HEIRDialect>();
  registry.insert<mlir::heir::EncryptedArithDialect>();

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

  mlir::PassPipelineRegistration<>(
      "heir-tosa-to-arith",
      "Run passes to lower TOSA models with stripped quant types to arithmetic",
      tosaPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
