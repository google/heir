#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CHEDDARBUFFERIZE_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CHEDDARBUFFERIZE_H_

#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

// Bufferizes a cheddar module with the upstream One-Shot pipeline and turns
// every buffer result into a caller-provided out-param (`bufferize.result`).
void buildCheddarBufferizationPipeline(OpPassManager& pm);

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CHEDDARBUFFERIZE_H_
