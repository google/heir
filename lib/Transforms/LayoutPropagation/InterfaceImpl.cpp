#include "lib/Transforms/LayoutPropagation/InterfaceImpl.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

namespace {

template <typename OpTy>
struct IndexTypesNeedNoLayoutImpl
    : public OperandLayoutRequirementOpInterface::ExternalModel<
          IndexTypesNeedNoLayoutImpl<OpTy>, OpTy> {
  bool operandRequiresLayout(Operation* op, unsigned operandIndex,
                             bool isSecret) const {
    bool isIndex = isa<IndexType>(op->getOperand(operandIndex).getType());
    return !isIndex || isSecret;
  }
};

// The kernel for tensor.insert can put cleartexts directly into a plaintext
// mask, so a layout is not required. Otherwise, a layout is required for the
// scalar.
struct InsertionLayoutRequirement
    : public OperandLayoutRequirementOpInterface::ExternalModel<
          InsertionLayoutRequirement, tensor::InsertOp> {
  bool operandRequiresLayout(Operation* op, unsigned operandIndex,
                             bool isSecret) const {
    if (!isSecret) {
      return operandIndex == tensor::InsertOp::odsIndex_dest;
    }

    return operandIndex == tensor::InsertOp::odsIndex_dest ||
           operandIndex == tensor::InsertOp::odsIndex_scalar;
  }
};

struct OnlyExtractionSourceNeedsLayout
    : public OperandLayoutRequirementOpInterface::ExternalModel<
          OnlyExtractionSourceNeedsLayout, tensor::ExtractOp> {
  bool operandRequiresLayout(Operation* op, unsigned operandIndex,
                             bool isSecret) const {
    return operandIndex == tensor::ExtractOp::odsIndex_tensor;
  }
};

struct InsertSliceLayoutRequirement
    : public OperandLayoutRequirementOpInterface::ExternalModel<
          InsertSliceLayoutRequirement, tensor::InsertSliceOp> {
  bool operandRequiresLayout(Operation* op, unsigned operandIndex,
                             bool isSecret) const {
    if (!isSecret) {
      return operandIndex == tensor::InsertSliceOp::odsIndex_dest;
    }

    return operandIndex == tensor::InsertSliceOp::odsIndex_dest ||
           operandIndex == tensor::InsertSliceOp::odsIndex_source;
  }
};

}  // namespace

void registerOperandLayoutRequirementOpInterface(DialectRegistry& registry) {
  registry.addExtension(
      +[](MLIRContext* ctx, tensor_ext::TensorExtDialect* dialect) {
        tensor_ext::RotateOp::attachInterface<
            IndexTypesNeedNoLayoutImpl<tensor_ext::RotateOp>>(*ctx);
      });
  registry.addExtension(+[](MLIRContext* ctx, tensor::TensorDialect* dialect) {
    tensor::InsertOp::attachInterface<InsertionLayoutRequirement>(*ctx);
    tensor::ExtractOp::attachInterface<OnlyExtractionSourceNeedsLayout>(*ctx);
    tensor::InsertSliceOp::attachInterface<InsertSliceLayoutRequirement>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
