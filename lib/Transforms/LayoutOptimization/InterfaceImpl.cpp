#include "lib/Transforms/LayoutOptimization/InterfaceImpl.h"

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

namespace mlir {
namespace heir {

using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

Hoister createTrivialHoister(Operation* op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    LayoutAttr outputLayout = cast<LayoutAttr>(convertLayoutOp.getToLayout());
    result.convertLayoutOp = convertLayoutOp;
    result.newInputLayouts =
        SmallVector<LayoutAttr>(op->getNumOperands(), outputLayout);
    result.newKernel = KernelName::Trivial;
    result.newOutputLayout = outputLayout;
    return result;
  };
}

}  // namespace heir
}  // namespace mlir
