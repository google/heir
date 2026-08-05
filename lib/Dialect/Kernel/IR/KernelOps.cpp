#include "lib/Dialect/Kernel/IR/KernelOps.h"

// IWYU pragma: begin_keep
#include <bit>

#include "mlir/include/mlir/IR/BuiltinOps.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"             // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#define GET_OP_CLASSES
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace mlir {
namespace heir {
namespace kernel {

int EvalChebyshevOp::getLevelsToDrop() { return 0; }

::mlir::OpOperand& EvalChebyshevOp::getOperandToReduce() {
  return getOperation()->getOpOperand(0);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
