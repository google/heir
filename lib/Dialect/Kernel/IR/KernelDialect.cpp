#include "lib/Dialect/Kernel/IR/KernelDialect.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/Kernel/IR/KernelOps.h"
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/Kernel/IR/KernelDialect.cpp.inc"

namespace mlir {
namespace heir {
namespace kernel {

void KernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
