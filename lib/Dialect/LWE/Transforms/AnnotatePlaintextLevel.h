#ifndef LIB_DIALECT_LWE_TRANSFORMS_ANNOTATEPLAINTEXTLEVEL_H_
#define LIB_DIALECT_LWE_TRANSFORMS_ANNOTATEPLAINTEXTLEVEL_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DECL_ANNOTATEPLAINTEXTLEVEL
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_TRANSFORMS_ANNOTATEPLAINTEXTLEVEL_H_
