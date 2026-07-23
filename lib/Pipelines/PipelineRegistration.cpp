#include "lib/Pipelines/PipelineRegistration.h"

#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h"
#include "lib/Dialect/RNS/Transforms/LowerConvertBasis.h"
#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h"
#include "lib/Transforms/ConvertSecretExtractToStaticExtract/ConvertSecretExtractToStaticExtract.h"
#include "lib/Transforms/ConvertSecretForToStaticFor/ConvertSecretForToStaticFor.h"
#include "lib/Transforms/ConvertSecretInsertToStaticInsert/ConvertSecretInsertToStaticInsert.h"
#include "lib/Transforms/ConvertSecretWhileToStaticFor/ConvertSecretWhileToStaticFor.h"
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"
#include "lib/Transforms/EmitCInterface/EmitCInterface.h"
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"      // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

using namespace mlir;
using namespace heir;
using mlir::func::FuncOp;

namespace mlir::heir {

void prepareForBufferize(OpPassManager& manager) {
  manager.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
  // Needed to lower affine.map and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(createLowerAffinePass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
}

namespace {
// Runs one-shot bufferization, but skips the (O(n^2)) in-place read-after-write
// analysis for every function whose name contains "__preprocessing".
//
// The stock pass only accepts an exact-match list of symbol names
// (`noAnalysisFuncFilter`), which the pipeline builder cannot know up front
// because the entry function name varies per model. So we compute the matching
// names from the actual module here and then run the stock pass verbatim (same
// options, so behavior is identical apart from which functions skip analysis).
struct PreprocessingAwareOneShotBufferizePass
    : public PassWrapper<PreprocessingAwareOneShotBufferizePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PreprocessingAwareOneShotBufferizePass)

  StringRef getArgument() const final {
    return "heir-preprocessing-aware-one-shot-bufferize";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    // Reuse exactly the stock pass's dependent dialects.
    bufferization::createOneShotBufferizePass()->getDependentDialects(registry);
  }

  void runOnOperation() override {
    SmallVector<std::string> noAnalysisFuncFilter;
    getOperation().walk([&](FuncOp funcOp) {
      if (funcOp.getSymName().contains("__preprocessing")) {
        noAnalysisFuncFilter.push_back(funcOp.getSymName().str());
      }
    });

    bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    bufferizationOptions.allowReturnAllocsFromLoops = true;
    bufferizationOptions.noAnalysisFuncFilter = noAnalysisFuncFilter;

    OpPassManager pipeline(ModuleOp::getOperationName());
    pipeline.addPass(
        bufferization::createOneShotBufferizePass(bufferizationOptions));
    if (failed(runPipeline(pipeline, getOperation()))) {
      signalPassFailure();
    }
  }
};
}  // namespace

void oneShotBufferize(OpPassManager& manager, bool includeDeallocation) {
  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  manager.addPass(std::make_unique<PreprocessingAwareOneShotBufferizePass>());
  manager.addPass(memref::createExpandReallocPass());
  if (includeDeallocation) {
    manager.addPass(
        bufferization::createOwnershipBasedBufferDeallocationPass());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(
        bufferization::createBufferDeallocationSimplificationPass());
    manager.addPass(bufferization::createLowerDeallocationsPass());
  }
  manager.addPass(createCSEPass());
  manager.addPass(mlir::createConvertBufferizationToMemRefPass());
  manager.addPass(createCanonicalizerPass());
}

void mathToPolynomialApproximationBuilder(OpPassManager& pm) {
  pm.addPass(createPolynomialApproximation());
  pm.addPass(createLowerPolynomialEval());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void polynomialToLLVMPipelineBuilder(OpPassManager& manager) {
  // Annotate public functions with a C interface
  manager.addPass(createEmitCInterface());

  // Poly
  manager.addPass(createLowerPolynomialEval());
  ElementwiseToAffineOptions elementwiseOptions;
  elementwiseOptions.convertDialects = {"polynomial"};
  manager.addPass(createElementwiseToAffine(elementwiseOptions));
  manager.addPass(polynomial::createPolynomialToModArith());
  manager.addPass(rns::createLowerConvertBasis());
  manager.addPass(::mlir::heir::mod_arith::createModArithToArith());
  manager.addPass(createCanonicalizerPass());

  // Bufferize
  prepareForBufferize(manager);
  oneShotBufferize(manager);

  // Linalg must be bufferized before it can be lowered
  // But lowering to loops also re-introduces affine.apply, so re-lower that
  manager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createConvertBufferizationToMemRefPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());

  // ToLLVM
  manager.addPass(arith::createArithExpandOpsPass());
  manager.addPass(createSCFToControlFlowPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  // expand strided metadata will create affine map. Needed to lower affine.map
  // and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createConvertToLLVMPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

void basicMLIRToLLVMPipelineBuilder(OpPassManager& manager) {
  // Bufferize
  prepareForBufferize(manager);
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
  manager.addPass(createSCFToControlFlowPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  manager.addPass(createConvertToLLVMPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

void convertToDataObliviousPipelineBuilder(OpPassManager& manager) {
  // Access Transformation
  manager.addPass(createConvertSecretExtractToStaticExtract());
  manager.addPass(createConvertSecretInsertToStaticInsert());

  // Loop Transformation
  manager.addPass(createConvertSecretWhileToStaticFor());
  manager.addPass(createConvertSecretForToStaticFor());

  // If Transformation
  manager.addPass(createConvertIfToSelect());
}

}  // namespace mlir::heir
