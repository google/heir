#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"

#include <cmath>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
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
  return binop<MultiplyNode<SSAValue>, ckks::MulOp, ckks::MulPlainOp>(node);
}

Value IRMaterializingVisitor::operator()(const LeftRotateNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  IntegerAttr shift = builder.getI64IntegerAttr(node.shift);
  return ckks::RotateOp::create(builder, operand.getType(), operand, shift);
}

Value IRMaterializingVisitor::operator()(const ExtractNode<SSAValue>& node) {
  // Unlike the version of IRMaterializingVisitor that operates before the
  // scheme level, here we are literally extracting one entry from a 1D tensor.
  Value operand = this->process(node.operand);
  Value index = arith::ConstantIndexOp::create(builder, node.index);
  return tensor::ExtractOp::create(builder, operand, index);
}

}  // namespace orion
}  // namespace heir
}  // namespace mlir
