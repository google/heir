#ifndef LIB_DIALECT_LATTIGO_TRANSFORMS_ALLOCTOINPLACE_H_
#define LIB_DIALECT_LATTIGO_TRANSFORMS_ALLOCTOINPLACE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_DECL_ALLOCTOINPLACE
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LATTIGO_TRANSFORMS_ALLOCTOINPLACE_H_
