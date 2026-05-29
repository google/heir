#ifndef LIB_PIPELINES_PIPELINEREGISTRATION_H_
#define LIB_PIPELINES_PIPELINEREGISTRATION_H_

#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project

namespace mlir::heir {

// Prepares IR for `oneShotBufferize` by wrapping elementwise tensor ops in
// `linalg.generic` and expanding any residual affine/memref ops that
// bufferization can't handle.
void prepareForBufferize(OpPassManager& manager);

void oneShotBufferize(OpPassManager& manager, bool includeDeallocation = true);

void mathToPolynomialApproximationBuilder(OpPassManager& pm);

void polynomialToLLVMPipelineBuilder(OpPassManager& manager);

void basicMLIRToLLVMPipelineBuilder(OpPassManager& manager);

void convertToDataObliviousPipelineBuilder(OpPassManager& manager);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_PIPELINEREGISTRATION_H_
