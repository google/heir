#ifndef LIB_DIALECT_SECRET_TRANSFORMS_FORGETSECRETS_H_
#define LIB_DIALECT_SECRET_TRANSFORMS_FORGETSECRETS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DECL_SECRETFORGETSECRETS
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_TRANSFORMS_FORGETSECRETS_H_
