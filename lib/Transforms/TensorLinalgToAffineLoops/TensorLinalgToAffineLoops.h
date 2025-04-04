#ifndef LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_H_
#define LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_H_
