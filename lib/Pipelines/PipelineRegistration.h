#ifndef LIB_PIPELINES_PIPELINEREGISTRATION_H_
#define LIB_PIPELINES_PIPELINEREGISTRATION_H_

#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project

namespace mlir::heir {

void tosaToLinalg(OpPassManager& manager);

void oneShotBufferize(OpPassManager& manager);

void mathToPolynomialApproximationBuilder(OpPassManager& pm);

void polynomialToLLVMPipelineBuilder(OpPassManager& manager);

void basicMLIRToLLVMPipelineBuilder(OpPassManager& manager);

void convertToDataObliviousPipelineBuilder(OpPassManager& manager);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_PIPELINEREGISTRATION_H_
