#ifndef LIB_DIALECT_POULPY_TRANSFORMS_POULPYBUFFERIZE_H_
#define LIB_DIALECT_POULPY_TRANSFORMS_POULPYBUFFERIZE_H_

#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poulpy {

// Bufferizes a poulpy module with the upstream One-Shot pipeline. Function
// results stay results, returned by value in the emitted Rust; a result that
// is equivalent to an argument is dropped and the argument updated in place.
void buildPoulpyBufferizationPipeline(OpPassManager& pm);

}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POULPY_TRANSFORMS_POULPYBUFFERIZE_H_
