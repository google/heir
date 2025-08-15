#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_HOISTING_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_HOISTING_H_

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

struct HoistResult {
  // A new required layout for each operand
  SmallVector<::mlir::Attribute> newInputLayouts;

  // A new result layout
  ::mlir::Attribute newOutputLayout;

  // A new result layout
  ::mlir::heir::KernelName newKernel;

  // The convert_layout op hoisted.
  ::mlir::heir::tensor_ext::ConvertLayoutOp convertLayoutOp;
};

// A Hoister maps convert_layout to HoistResult, returning failure if the hoist
// is impossible.
using Hoister = std::function<::llvm::FailureOr<::mlir::heir::HoistResult>(
    ::mlir::heir::tensor_ext::ConvertLayoutOp)>;

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_HOISTING_H_
