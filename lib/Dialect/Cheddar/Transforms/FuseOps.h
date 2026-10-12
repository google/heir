#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DECL
#include "lib/Dialect/Cheddar/Transforms/FuseOps.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Cheddar/Transforms/FuseOps.h.inc"

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_H_
