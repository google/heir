#ifndef LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Kernel/IR/KernelDialect.h"
#include "lib/Dialect/Kernel/Transforms/PrepareLinearTransforms.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace kernel {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Kernel/Transforms/Passes.h.inc"

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_H_
