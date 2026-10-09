#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTROTATEANDREDUCE_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTROTATEANDREDUCE_H_

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DECL_IMPLEMENTROTATEANDREDUCE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

LogicalResult convertRotateAndReduceOp(RotateAndReduceOp op,
                                       bool unroll = true);

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTROTATEANDREDUCE_H_
