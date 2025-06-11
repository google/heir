#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_HOISTRESULT_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_HOISTRESULT_H_

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"

namespace mlir {
namespace heir {

struct HoistResult {
  // A new required layout for each operand
  SmallVector<::mlir::heir::tensor_ext::LayoutAttr> newInputLayouts;

  // A new result layout
  ::mlir::heir::tensor_ext::LayoutAttr newOutputLayout;

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

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_HOISTRESULT_H_
