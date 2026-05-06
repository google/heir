#ifndef LIB_DIALECT_RNS_TRANSFORMS_LOWERCONVERTBASIS_H_
#define LIB_DIALECT_RNS_TRANSFORMS_LOWERCONVERTBASIS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

#define GEN_PASS_DECL_LOWERCONVERTBASIS
#include "lib/Dialect/RNS/Transforms/Passes.h.inc"

}  // namespace rns
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_RNS_TRANSFORMS_LOWERCONVERTBASIS_H_
