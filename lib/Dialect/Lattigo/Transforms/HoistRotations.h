#ifndef LIB_DIALECT_LATTIGO_TRANSFORMS_HOISTROTATIONS_H_
#define LIB_DIALECT_LATTIGO_TRANSFORMS_HOISTROTATIONS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_DECL_HOISTROTATIONS
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createLattigoHoistRotations();

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LATTIGO_TRANSFORMS_HOISTROTATIONS_H_
