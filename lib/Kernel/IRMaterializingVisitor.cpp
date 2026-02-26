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

namespace {

// Helper to convert DagType to MLIR Type
Type dagTypeToMLIRType(const DagType& dagType, OpBuilder& builder) {
  return std::visit(
      [&](auto&& arg) -> Type {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, kernel::IntegerType>) {
          return builder.getIntegerType(arg.bitWidth);
        } else if constexpr (std::is_same_v<T, kernel::FloatType>) {
          if (arg.bitWidth == 32) {
            return builder.getF32Type();
          } else if (arg.bitWidth == 64) {
            return builder.getF64Type();
          } else if (arg.bitWidth == 16) {
            return builder.getF16Type();
          } else {
            llvm_unreachable("Unsupported float bit width");
          }
        } else if constexpr (std::is_same_v<T, kernel::IndexType>) {
          return builder.getIndexType();
        } else if constexpr (std::is_same_v<T, kernel::IntTensorType>) {
          auto elementType = builder.getIntegerType(arg.bitWidth);
          return RankedTensorType::get(arg.shape, elementType);
        } else if constexpr (std::is_same_v<T, kernel::FloatTensorType>) {
          mlir::Type elementType;
          if (arg.bitWidth == 32) {
            elementType = builder.getF32Type();
          } else if (arg.bitWidth == 64) {
            elementType = builder.getF64Type();
          } else if (arg.bitWidth == 16) {
            elementType = builder.getF16Type();
          } else {
            llvm_unreachable("Unsupported float bit width");
          }
          return RankedTensorType::get(arg.shape, elementType);
        }
        llvm_unreachable("Unknown DagType variant");
      },
      dagType.type_variant);
}

}  // namespace

std::vector<Value> IRMaterializingVisitor::operator()(
    const LeafNode<SSAValue>& node) {
  return {node.value.getValue()};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ConstantScalarNode& node) {
  Type targetType = dagTypeToMLIRType(node.type, builder);
  TypedAttr attr;
  if (auto indexTy = dyn_cast<mlir::IndexType>(targetType)) {
    // Handle index type specially - convert double to index integer
    APInt apVal = APInt(64, static_cast<int64_t>(std::floor(node.value)));
    attr = builder.getIntegerAttr(indexTy, apVal);
  } else if (auto floatTy = dyn_cast<mlir::FloatType>(targetType)) {
    APFloat apVal(node.value);
    APFloat converted =
        convertFloatToSemantics(apVal, floatTy.getFloatSemantics());
    attr = builder.getFloatAttr(floatTy, converted);
  } else if (auto intTy = dyn_cast<mlir::IntegerType>(targetType)) {
    int64_t intValue = static_cast<int64_t>(std::floor(node.value));
    APInt apVal(intTy.getWidth(), intValue, /*isSigned=*/true);
    attr = builder.getIntegerAttr(intTy, apVal);
  }

  auto constantOp = arith::ConstantOp::create(builder, targetType, attr);
  createdOpCallback(constantOp);
  return {constantOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(const SplatNode& node) {
  Type targetType = dagTypeToMLIRType(node.type, builder);

  // SplatNode should always produce a tensor type
  RankedTensorType tensorTy = cast<RankedTensorType>(targetType);
  Type elementType = tensorTy.getElementType();

  TypedAttr attr;
  if (auto floatTy = dyn_cast<mlir::FloatType>(elementType)) {
    APFloat apVal(node.value);
    APFloat converted =
        convertFloatToSemantics(apVal, floatTy.getFloatSemantics());
    attr = DenseElementsAttr::get(tensorTy, converted);
  } else if (auto intTy = dyn_cast<mlir::IntegerType>(elementType)) {
    int64_t intValue = static_cast<int64_t>(std::floor(node.value));
    APInt apVal(intTy.getWidth(), intValue, /*isSigned=*/true);
    attr = DenseElementsAttr::get(tensorTy, apVal);
  } else {
    llvm_unreachable("Unsupported element type for SplatNode");
  }

  auto constantOp = arith::ConstantOp::create(builder, targetType, attr);
  createdOpCallback(constantOp);
  return {constantOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ConstantTensorNode& node) {
  Type targetType = dagTypeToMLIRType(node.type, builder);
  RankedTensorType tensorTy = cast<RankedTensorType>(targetType);

  TypedAttr attr;
  if (auto floatTy = dyn_cast<mlir::FloatType>(tensorTy.getElementType())) {
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
    auto intTy = cast<mlir::IntegerType>(tensorTy.getElementType());
    SmallVector<APInt> values;
    for (double v : node.value) {
      APInt apVal(intTy.getWidth(), std::floor(v));
      values.push_back(apVal);
    }
    attr = DenseElementsAttr::get(tensorTy, values);
  }

  auto constantOp = arith::ConstantOp::create(builder, targetType, attr);
  createdOpCallback(constantOp);
  return {constantOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const AddNode<SSAValue>& node) {
  return binop<AddNode<SSAValue>, arith::AddFOp, arith::AddIOp>(node);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const SubtractNode<SSAValue>& node) {
  return binop<SubtractNode<SSAValue>, arith::SubFOp, arith::SubIOp>(node);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const MultiplyNode<SSAValue>& node) {
  return binop<MultiplyNode<SSAValue>, arith::MulFOp, arith::MulIOp>(node);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const FloorDivNode<SSAValue>& node) {
  Value lhs = this->process(node.left)[0];
  Value rhs =
      arith::ConstantIntOp::create(builder, lhs.getType(), node.divisor);
  auto op = arith::DivSIOp::create(builder, lhs, rhs);
  createdOpCallback(op);
  return {op->getResult(0)};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const LeftRotateNode<SSAValue>& node) {
  Value operand = this->process(node.operand)[0];
  Value shift = this->process(node.shift)[0];
  auto rotateOp =
      tensor_ext::RotateOp::create(builder, operand.getType(), operand, shift);
  createdOpCallback(rotateOp);
  return {rotateOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ExtractNode<SSAValue>& node) {
  Value operand = this->process(node.operand)[0];
  Value index = arith::ConstantIndexOp::create(builder, node.index);

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
    auto extractOp = tensor::ExtractSliceOp::create(builder, tensorTy, operand,
                                                    offsets, sizes, strides);
    createdOpCallback(extractOp);
    return {extractOp};
  }

  // Otherwise let the type be inferred, though this will likely result in an
  // issue because the row index is preserved in the result type
  auto extractOp =
      tensor::ExtractSliceOp::create(builder, operand, offsets, sizes, strides);
  createdOpCallback(extractOp);
  return {extractOp};
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
