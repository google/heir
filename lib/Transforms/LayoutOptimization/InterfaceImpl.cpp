
#include "lib/Transforms/LayoutOptimization/Hoisting.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project

namespace mlir {
namespace heir {

using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

Hoister createTrivialHoister(Operation* op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    auto outputLayout = convertLayoutOp.getToLayout();
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
