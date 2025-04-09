#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ASSIGNLAYOUT_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ASSIGNLAYOUT_H_

#include <cstdint>
#include <functional>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

// Lower tensor_ext.assign_layout. Returns the final value produced by the
// encoding implementation. Applies createdOpCallback to each created
// operation.
FailureOr<Value> implementAssignLayout(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ASSIGNLAYOUT_H_
