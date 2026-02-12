#include "lib/Kernel/IRMaterializingVisitor.h"

#include <cmath>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

Value IRMaterializingVisitor::process(
    const std::shared_ptr<ArithmeticDagNode<SSAValue>>& node, Type type) {
  assert(node != nullptr && "invalid null node!");

  // Use provided type, or fall back to evaluatedType
  Type targetType = type ? type : evaluatedType;

  // Check cache with (node, type) as key
  const auto* nodePtr = node.get();
  auto cacheKey = std::make_pair(nodePtr, targetType);
  if (auto it = typeAwareCache.find(cacheKey); it != typeAwareCache.end()) {
    return it->second;
  }

  // Set currentType for the visitor methods to use
  Type savedType = currentType;
  currentType = targetType;

  // Visit the node
  Value result = std::visit(*this, node->node_variant);

  // Restore previous currentType
  currentType = savedType;

  // Cache the result
  typeAwareCache[cacheKey] = result;
  return result;
}

Value IRMaterializingVisitor::operator()(const LeafNode<SSAValue>& node) {
  return node.value.getValue();
}

Value IRMaterializingVisitor::operator()(const ConstantScalarNode& node) {
  // A "ConstantScalarNode" can still have a shaped type, and in that case the
  // value is treated as a splat. This is preferred in some cases to support
  // DAGs that can be evaluated elementwise for ElementwiseMappable ops like
  // arith ops.
  TypedAttr attr;
  if (auto indexTy = dyn_cast<IndexType>(getElementTypeOrSelf(currentType))) {
    // Handle index type specially - convert double to index integer
    APInt apVal = APInt(64, static_cast<int64_t>(std::floor(node.value)));
    attr = builder.getIntegerAttr(indexTy, apVal);
  } else if (auto floatTy =
                 dyn_cast<FloatType>(getElementTypeOrSelf(currentType))) {
    APFloat apVal(node.value);
    APFloat converted =
        convertFloatToSemantics(apVal, floatTy.getFloatSemantics());
    attr = getScalarOrDenseAttr(currentType, converted);
  } else if (auto intTy =
                 dyn_cast<IntegerType>(getElementTypeOrSelf(currentType))) {
    // Node values are doubles and we may have to properly support integers.
    APInt apVal(intTy.getWidth(), std::floor(node.value));
    attr = getScalarOrDenseAttr(currentType, apVal);
  }

  auto constantOp = arith::ConstantOp::create(builder, currentType, attr);
  createdOpCallback(constantOp);
  return constantOp;
}

Value IRMaterializingVisitor::operator()(const ConstantTensorNode& node) {
  RankedTensorType tensorTy = cast<RankedTensorType>(currentType);
  TypedAttr attr;
  if (auto floatTy = dyn_cast<FloatType>(tensorTy.getElementType())) {
    SmallVector<APFloat> values;
    for (double v : node.value) {
      APFloat apVal(v);
      APFloat converted =
          convertFloatToSemantics(apVal, floatTy.getFloatSemantics());
      values.push_back(converted);
    }
    attr = DenseElementsAttr::get(tensorTy, values);
  } else {
    // Node values are doubles and we must convert them to integers
    auto intTy = cast<IntegerType>(tensorTy.getElementType());
    SmallVector<APInt> values;
    for (double v : node.value) {
      APInt apVal(intTy.getWidth(), std::floor(v));
      values.push_back(apVal);
    }
    attr = DenseElementsAttr::get(tensorTy, values);
  }

  auto constantOp = arith::ConstantOp::create(builder, currentType, attr);
  createdOpCallback(constantOp);
  return constantOp;
}

Value IRMaterializingVisitor::operator()(const AddNode<SSAValue>& node) {
  return binop<AddNode<SSAValue>, arith::AddFOp, arith::AddIOp>(node);
}

Value IRMaterializingVisitor::operator()(const SubtractNode<SSAValue>& node) {
  return binop<SubtractNode<SSAValue>, arith::SubFOp, arith::SubIOp>(node);
}

Value IRMaterializingVisitor::operator()(const MultiplyNode<SSAValue>& node) {
  return binop<MultiplyNode<SSAValue>, arith::MulFOp, arith::MulIOp>(node);
}

Value IRMaterializingVisitor::operator()(const DivideNode<SSAValue>& node) {
  return binop<DivideNode<SSAValue>, arith::DivFOp, arith::DivSIOp>(node);
}

Value IRMaterializingVisitor::operator()(const PowerNode<SSAValue>& node) {
  assert(false && "PowerNode materialization not implemented");
  return Value();
}

Value IRMaterializingVisitor::operator()(const LeftRotateNode<SSAValue>& node) {
  Value operand = process(node.operand, currentType);
  // Shift should be materialized as an index type
  Value shift = process(node.shift, builder.getIndexType());
  auto rotateOp =
      tensor_ext::RotateOp::create(builder, currentType, operand, shift);
  createdOpCallback(rotateOp);
  return rotateOp;
}

Value IRMaterializingVisitor::operator()(const ExtractNode<SSAValue>& node) {
  // Process the operand with its natural type (nullptr = use
  // evaluatedType/currentType)
  Value operand = process(node.operand, nullptr);
  // Index should be materialized as an index type, not the tensor type
  Value index = process(node.index, builder.getIndexType());

  RankedTensorType tensorType = cast<RankedTensorType>(operand.getType());

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

  if (auto tensorTy = dyn_cast<RankedTensorType>(currentType)) {
    auto extractOp = tensor::ExtractSliceOp::create(builder, tensorTy, operand,
                                                    offsets, sizes, strides);
    createdOpCallback(extractOp);
    return extractOp;
  }

  // Otherwise let the type be inferred, though this will likely result in an
  // issue because the row index is preserved in the result type
  auto extractOp =
      tensor::ExtractSliceOp::create(builder, operand, offsets, sizes, strides);
  createdOpCallback(extractOp);
  return extractOp;
}

Value IRMaterializingVisitor::operator()(const VariableNode<SSAValue>& node) {
  assert(false && "VariableNode materialization not implemented");
  return Value();
}

Value IRMaterializingVisitor::operator()(const ForLoopNode<SSAValue>& node) {
  assert(false && "ForLoopNode materialization not implemented");
  return Value();
}

Value IRMaterializingVisitor::operator()(const YieldNode<SSAValue>& node) {
  assert(false && "YieldNode materialization not implemented");
  return Value();
}

Value IRMaterializingVisitor::operator()(const ResultAtNode<SSAValue>& node) {
  assert(false && "ResultAtNode materialization not implemented");
  return Value();
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
