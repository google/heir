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
  bool operandRequiresLayout(Operation* op, unsigned operandIndex) const {
    bool isIndex = isa<IndexType>(op->getOperand(operandIndex).getType());
    return !isIndex;
  }
};

// The kernel for tensor.insert can put cleartexts directly into a plaintext
// mask, so a layout is not required.
struct OnlyInsertionDestNeedsLayout
    : public OperandLayoutRequirementOpInterface::ExternalModel<
          OnlyInsertionDestNeedsLayout, tensor::InsertOp> {
  bool operandRequiresLayout(Operation* op, unsigned operandIndex) const {
    return operandIndex == tensor::InsertOp::odsIndex_dest;
  }
};

// The kernel for tensor.insert can put cleartexts directly into a plaintext
// mask, so a layout is not required.
struct OnlyExtractionSourceNeedsLayout
    : public OperandLayoutRequirementOpInterface::ExternalModel<
          OnlyExtractionSourceNeedsLayout, tensor::ExtractOp> {
  bool operandRequiresLayout(Operation* op, unsigned operandIndex) const {
    return operandIndex == tensor::ExtractOp::odsIndex_tensor;
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
    tensor::InsertOp::attachInterface<OnlyInsertionDestNeedsLayout>(*ctx);
    tensor::ExtractOp::attachInterface<OnlyExtractionSourceNeedsLayout>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
