#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_INSERTROTATE_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_INSERTROTATE_H_

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DECL_INSERTROTATE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_INSERTROTATE_H_
