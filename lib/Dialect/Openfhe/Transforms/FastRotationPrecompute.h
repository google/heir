#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_FASTROTATIONPRECOMPUTE_H_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_FASTROTATIONPRECOMPUTE_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DECL_FASTROTATIONPRECOMPUTE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_FASTROTATIONPRECOMPUTE_H_
