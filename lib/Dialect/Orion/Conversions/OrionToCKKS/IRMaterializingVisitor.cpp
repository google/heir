#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"

#include <cmath>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace orion {

using kernel::AddNode;
using kernel::ConstantScalarNode;
using kernel::ConstantTensorNode;
using kernel::ExtractNode;
using kernel::LeafNode;
using kernel::LeftRotateNode;
using kernel::MultiplyNode;
using kernel::SSAValue;
using kernel::SubtractNode;

Value IRMaterializingVisitor::operator()(const LeafNode<SSAValue>& node) {
  return node.value.getValue();
}

Value IRMaterializingVisitor::operator()(const ConstantScalarNode& node) {
  llvm_unreachable("not supported");
  return Value();
}

Value IRMaterializingVisitor::operator()(const ConstantTensorNode& node) {
  llvm_unreachable("not supported");
  return Value();
}

Value IRMaterializingVisitor::operator()(const AddNode<SSAValue>& node) {
  return binop<AddNode<SSAValue>, ckks::AddOp, ckks::AddPlainOp>(node);
}

Value IRMaterializingVisitor::operator()(const SubtractNode<SSAValue>& node) {
  return binop<SubtractNode<SSAValue>, ckks::SubOp, ckks::SubPlainOp>(node);
}

Value IRMaterializingVisitor::operator()(const MultiplyNode<SSAValue>& node) {
  return binop<MultiplyNode<SSAValue>, ckks::MulOp, ckks::MulPlainOp>(
      node,
      /*rescale=*/true);
}

Value IRMaterializingVisitor::operator()(const LeftRotateNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
    // The thing being rotated is a ciphertext-semantic tensor
    Value shift = arith::ConstantIntOp::create(builder, node.shift, 64);
    return tensor_ext::RotateOp::create(builder, operand.getType(), operand,
                                        shift);
  }

  IntegerAttr shift = builder.getI64IntegerAttr(node.shift);
  return ckks::RotateOp::create(builder, operand, shift);
}

Value IRMaterializingVisitor::operator()(const ExtractNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  RankedTensorType tensorType = cast<RankedTensorType>(operand.getType());
  Value index = arith::ConstantIndexOp::create(builder, node.index);
  if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(tensorType))) {
    return tensor::ExtractOp::create(builder, operand, index);
  }

  // Extracting 1 row of a matrix, so offset is 0 except for the row dim
  SmallVector<OpFoldResult> offsets(tensorType.getRank(),
                                    builder.getIndexAttr(0));
  offsets[0] = index;

  // Sizes are 1 in the row dim, and the full size in other dims
  SmallVector<OpFoldResult> sizes;
  sizes.push_back(builder.getIndexAttr(1));
  for (int i = 1; i < tensorType.getRank(); ++i) {
    sizes.push_back(builder.getIndexAttr(tensorType.getDimSize(i)));
  }

  // Strides are all 1
  SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                    builder.getIndexAttr(1));

  RankedTensorType extractedType = RankedTensorType::get(
      tensorType.getShape().drop_front(), tensorType.getElementType());
  return builder.create<tensor::ExtractSliceOp>(extractedType, operand, offsets,
                                                sizes, strides);
}

}  // namespace orion
}  // namespace heir
}  // namespace mlir
