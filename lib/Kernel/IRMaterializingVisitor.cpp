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

Value IRMaterializingVisitor::operator()(const LeafNode<SSAValue>& node) {
  return node.value.getValue();
}

Value IRMaterializingVisitor::operator()(const ConstantScalarNode& node) {
  // A "ConstantScalarNode" can still have a shaped type, and in that case the
  // value is treated as a splat. This is preferred in some cases to support
  // DAGs that can be evaluated elementwise for ElementwiseMappable ops like
  // arith ops.
  TypedAttr attr;
  if (auto floatTy = dyn_cast<FloatType>(getElementTypeOrSelf(evaluatedType))) {
    APFloat apVal(node.value);
    APFloat converted =
        convertFloatToSemantics(apVal, floatTy.getFloatSemantics());
    attr = getScalarOrDenseAttr(evaluatedType, converted);
  } else if (auto intTy =
                 dyn_cast<IntegerType>(getElementTypeOrSelf(evaluatedType))) {
    // Node values are doubles and we may have to properly support integers.
    APInt apVal(intTy.getWidth(), std::floor(node.value));
    attr = getScalarOrDenseAttr(evaluatedType, apVal);
  }

  auto constantOp = arith::ConstantOp::create(builder, evaluatedType, attr);
  createdOpCallback(constantOp);
  return constantOp;
}

Value IRMaterializingVisitor::operator()(const ConstantTensorNode& node) {
  RankedTensorType tensorTy = cast<RankedTensorType>(evaluatedType);
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

  auto constantOp = arith::ConstantOp::create(builder, evaluatedType, attr);
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

Value IRMaterializingVisitor::operator()(const LeftRotateNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  Value shift = builder.create<arith::ConstantIndexOp>(node.shift);
  auto rotateOp =
      builder.create<tensor_ext::RotateOp>(evaluatedType, operand, shift);
  createdOpCallback(rotateOp);
  return rotateOp;
}

Value IRMaterializingVisitor::operator()(const ExtractNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  Value index = builder.create<arith::ConstantIndexOp>(node.index);

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

  if (auto tensorTy = dyn_cast<RankedTensorType>(evaluatedType)) {
    auto extractOp = builder.create<tensor::ExtractSliceOp>(
        tensorTy, operand, offsets, sizes, strides);
    createdOpCallback(extractOp);
    return extractOp;
  }

  // Otherwise let the type be inferred, though this will likely result in an
  // issue because the row index is preserved in the result type
  auto extractOp =
      builder.create<tensor::ExtractSliceOp>(operand, offsets, sizes, strides);
  createdOpCallback(extractOp);
  return extractOp;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
