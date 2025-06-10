
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"  // from @llvm-project
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project

namespace mlir {
namespace heir {

using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;
static auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;

static Hoister createTrivialHoister(Operation* op) {
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

static Hoister createTrailingDimPrecomposingMatvecHoister(linalg::MatvecOp op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    auto fromLayout = convertLayoutOp.getFromLayout();
    auto toLayout = convertLayoutOp.getToLayout();

    result.convertLayoutOp = convertLayoutOp;
    // All the matvec kernels we have today should maintain the layout of the
    // vector before and after the op.
    result.newOutputLayout = toLayout;

    // The kernel is unchanged, so copy the existing kernel attr
    result.newKernel = op->getAttrOfType<secret::KernelAttr>(
                             secret::SecretDialect::kKernelAttrName)
                           .getName();

    // Operand order for Matvec op is:
    //
    // 0: matrix
    // 1: input vector
    // 2: output vector
    FailureOr<Attribute> oldMatrixLayoutRes =
        findAttributeAssociatedWith(op->getOperand(0), kLayoutAttrName);
    assert(succeeded(oldMatrixLayoutRes) && "failed to find matrix layout!");
    LayoutAttr oldMatrixLayout = cast<LayoutAttr>(oldMatrixLayoutRes.value());

    AffineMap incrementalVecLayoutChange = toLayout.compose(...);

    AffineMap newMatrixMap = AffineMap::compose(oldMatrixLayout, );
    // Alignment is assumed to be the same as a precondition on this function
    LayoutAttr newMatrixLayout =
        LayoutAttr::get(map, oldMatrixLayout.getAlignment());
    result.newInputLayouts =
        SmallVector<LayoutAttr>{newMatrixLayout, toLayout, toLayout};
    return result;
  };
}

}  // namespace heir
}  // namespace mlir
