#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project

namespace mlir {
namespace heir {

/// Construct a trivial hoister for which all layouts can be hoisted without
/// any kernel change or difference in the output layout.
Hoister createTrivialHoister(Operation* op);

/// Construct a hoister that pre-composes the last dimension of the Matvec op's
/// matrix with the incremental transformation required to go from vecFromLayout
/// to vecToLayout, while keeping the kernel the same. Requires the alignment
/// attribute to be unchanged in the context layout conversion being hoisted.
Hoister createTrailingDimPrecomposingMatvecHoister(linalg::MatvecOp op);

template <typename OpTy>
struct DoNothingHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          DoNothingHoistingImpl<OpTy>, OpTy> {
  std::vector<::mlir::heir::Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    return {createTrivialHoister(op)};
  }
};

struct MatvecHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          MatvecHoistingImpl<linalg::MatvecOp>, linalg::MatvecOp> {
  std::vector<Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    std::vector<Hoister> hoisters;
    linalg::MatvecOp matvecOp = cast<linalg::MatvecOp>(op);

    auto kernel = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (!kernel) return hositers;

    if (!op->getAttrOfType<tensor_ext::LayoutAttr>(
            tensor_ext::TensorExtDialect::kLayoutAttrName))
      return hoisters;

    auto fromLayout = convertLayoutOp.getFromLayout();
    auto toLayout = convertLayoutOp.getToLayout();
    if (fromLayout.getAlignment() != toLayout.getAlignment()) return hoisters;

    switch (kernel.getName()) {
      case heir::KernelName::MatvecNaive:
      case heir::KernelName::MatvecDiagonal:
        hoisters.push_back(createTrailingDimPrecomposingMatvecHoister(
            matvecOp, fromLayout, toLayout));
        break;
      default:
        assert(false && "unknown kernel");
        break;
    }

    return hoisters;
  }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
