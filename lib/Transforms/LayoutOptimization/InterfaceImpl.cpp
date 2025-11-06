#include "lib/Transforms/LayoutOptimization/InterfaceImpl.h"

#include <cassert>
#include <vector>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "interface-impl"

namespace mlir {
namespace heir {

using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;
static auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
using ::mlir::linalg::MatvecOp;

namespace {

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
          MatvecHoistingImpl, MatvecOp> {
  std::vector<Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    std::vector<Hoister> hoisters;
    linalg::MatvecOp matvecOp = cast<linalg::MatvecOp>(op);

    auto kernel = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (!kernel) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Kernel attribute not found on op " << *op << "\n");
      return hoisters;
    }

    if (!op->hasAttr(tensor_ext::TensorExtDialect::kLayoutAttrName)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Layout attribute not found on op " << *op << "\n");
      return hoisters;
    }

    switch (kernel.getName()) {
      case heir::KernelName::MatvecNaive:
      case heir::KernelName::MatvecDiagonal:
        hoisters.push_back(createPrecomposingMatvecHoister(matvecOp));
        break;
      default:
        assert(false && "unsupported kernel for layout hoisting");
        break;
    }

    return hoisters;
  }
};

struct MatmulHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          MatmulHoistingImpl, linalg::MatmulOp> {
  std::vector<Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    std::vector<Hoister> hoisters;

    auto kernel = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (!kernel) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Kernel attribute not found on op " << *op << "\n");
      return hoisters;
    }

    if (!op->hasAttr(tensor_ext::TensorExtDialect::kLayoutAttrName)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Layout attribute not found on op " << *op << "\n");
      return hoisters;
    }

    // TODO(#2385): try hoisting layout conversion through bicyclic matmul
    return hoisters;
  }
};

}  // namespace

Hoister createTrivialHoister(Operation* op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    Attribute outputLayout = convertLayoutOp.getToLayout();
    result.convertLayoutOp = convertLayoutOp;
    result.newInputLayouts =
        SmallVector<Attribute>(op->getNumOperands(), outputLayout);
    result.newKernel = KernelName::Trivial;
    result.newOutputLayout = outputLayout;
    return result;
  };
}

Hoister createPrecomposingMatvecHoister(linalg::MatvecOp op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    auto fromLayout = dyn_cast<LayoutAttr>(convertLayoutOp.getFromLayout());
    auto toLayout = dyn_cast<LayoutAttr>(convertLayoutOp.getToLayout());

    if (!fromLayout || !toLayout) return failure();

    // Operand order for Matvec op is:
    //
    // 0: matrix
    // 1: input vector
    // 2: output vector
    FailureOr<Attribute> oldMatrixLayoutRes =
        findAttributeAssociatedWith(op->getOperand(0), kLayoutAttrName);
    assert(succeeded(oldMatrixLayoutRes) && "failed to find matrix layout!");
    LayoutAttr oldMatrixLayout =
        dyn_cast<LayoutAttr>(oldMatrixLayoutRes.value());
    if (!oldMatrixLayout) return failure();

    result.convertLayoutOp = convertLayoutOp;
    // All the matvec kernels we have today should maintain the layout of the
    // vector before and after the op.
    result.newOutputLayout = toLayout;

    // The kernel is unchanged, so copy the existing kernel attr
    result.newKernel = op->getAttrOfType<secret::KernelAttr>(
                             secret::SecretDialect::kKernelAttrName)
                           .getName();

    presburger::IntegerRelation newMatrixLayoutRelation =
        hoistConversionThroughMatvec(oldMatrixLayout.getIntegerRelation(),
                                     fromLayout.getIntegerRelation(),
                                     toLayout.getIntegerRelation());
    Attribute newMatrixLayout = LayoutAttr::getFromIntegerRelation(
        op->getContext(), newMatrixLayoutRelation);
    result.newInputLayouts =
        SmallVector<Attribute>{newMatrixLayout, toLayout, toLayout};
    return result;
  };
}

void registerLayoutConversionHoistableInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect* dialect) {
    arith::AddFOp::attachInterface<DoNothingHoistingImpl<arith::AddFOp>>(*ctx);
    arith::AddIOp::attachInterface<DoNothingHoistingImpl<arith::AddIOp>>(*ctx);
    arith::MulFOp::attachInterface<DoNothingHoistingImpl<arith::MulFOp>>(*ctx);
    arith::MulIOp::attachInterface<DoNothingHoistingImpl<arith::MulIOp>>(*ctx);
    arith::SubFOp::attachInterface<DoNothingHoistingImpl<arith::SubFOp>>(*ctx);
    arith::SubIOp::attachInterface<DoNothingHoistingImpl<arith::SubIOp>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect* dialect) {
    linalg::MatvecOp::attachInterface<MatvecHoistingImpl>(*ctx);
    linalg::MatmulOp::attachInterface<MatmulHoistingImpl>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
