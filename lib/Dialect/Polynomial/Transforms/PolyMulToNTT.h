#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POLYMULTONTT_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POLYMULTONTT_H_

// IWYU pragma: begin_keep
#include <memory>

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL_POLYMULTONTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POLYMULTONTT_H_
