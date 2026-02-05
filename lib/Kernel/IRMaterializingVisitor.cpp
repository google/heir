#include "lib/Kernel/IRMaterializingVisitor.h"

#include <cmath>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
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
          // MLIR has specific methods for common float types
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

Value IRMaterializingVisitor::ensureIndexType(Value val) {
  if (val.getType().isIndex()) return val;
  return builder.create<arith::IndexCastOp>(builder.getIndexType(), val);
}

std::vector<Value> IRMaterializingVisitor::processInternal(
    const std::shared_ptr<ArithmeticDagNode<SSAValue>>& node) {
  assert(node != nullptr && "invalid null node!");

  const auto* nodePtr = node.get();
  if (auto it = cache.find(nodePtr); it != cache.end()) {
    return it->second;
  }

  std::vector<Value> result = std::visit(*this, node->node_variant);
  cache[nodePtr] = result;
  return result;
}

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

std::vector<Value> IRMaterializingVisitor::operator()(
    const SplatNode& node) {
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
    const DivideNode<SSAValue>& node) {
  return binop<DivideNode<SSAValue>, arith::DivFOp, arith::DivSIOp>(node);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const PowerNode<SSAValue>& node) {
  assert(false && "PowerNode materialization not implemented");
  return {Value()};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const LeftRotateNode<SSAValue>& node) {
  Value operand = processInternal(node.operand)[0];
  // Shift gets natural type, then cast to index if needed
  Value shift = processInternal(node.shift)[0];
  shift = ensureIndexType(shift);
  auto rotateOp =
      tensor_ext::RotateOp::create(builder, operand.getType(), operand, shift);
  createdOpCallback(rotateOp);
  return {rotateOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ExtractNode<SSAValue>& node) {
  // Process the operand with its natural type
  Value operand = processInternal(node.operand)[0];
  // Index gets natural type, then cast to index if needed
  Value index = processInternal(node.index)[0];
  index = ensureIndexType(index);

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

  // Infer the result type by taking a slice
  auto extractOp =
      tensor::ExtractSliceOp::create(builder, operand, offsets, sizes, strides);
  createdOpCallback(extractOp);

  // If we extracted a single row from a matrix (resulting in shape [1, N, ...]),
  // collapse the singleton dimension to get shape [N, ...]
  Value result = extractOp;
  RankedTensorType extractedType = cast<RankedTensorType>(result.getType());
  if (extractedType.getRank() > 1 && extractedType.getDimSize(0) == 1) {
    SmallVector<int64_t> newShape(extractedType.getShape().drop_front());
    RankedTensorType collapsedType =
        RankedTensorType::get(newShape, extractedType.getElementType());

    SmallVector<ReassociationIndices> reassociation;
    // Collapse first dimension with the second
    reassociation.push_back({0, 1});
    // Keep remaining dimensions as-is
    for (int i = 2; i < extractedType.getRank(); ++i) {
      reassociation.push_back({i});
    }

    auto collapseOp = tensor::CollapseShapeOp::create(
        builder, collapsedType, result, reassociation);
    createdOpCallback(collapseOp);
    result = collapseOp;
  }

  return {result};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const VariableNode<SSAValue>& node) {
  assert(node.value.has_value() && "VariableNode value is not set");
  return {node.value->getValue()};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ForLoopNode<SSAValue>& node) {
  std::vector<Value> initValues;
  initValues.reserve(node.inits.size());
  for (const auto& init : node.inits) {
    Value initProcessed = processInternal(init)[0];
    initValues.push_back(initProcessed);
  }

  Value lower = arith::ConstantIntOp::create(builder, node.lower, 32);
  Value upper = arith::ConstantIntOp::create(builder, node.upper, 32);
  Value step = arith::ConstantIntOp::create(builder, node.step, 32);
  auto loop = scf::ForOp::create(
      builder, lower, upper, step, initValues,
      [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iv,
          ValueRange args) {
        auto& inductionVarNode =
            std::get<VariableNode<SSAValue>>(node.inductionVar->node_variant);
        inductionVarNode.value = iv;

        assert(node.iterArgs.size() == args.size());
        for (size_t j = 0; j < args.size(); ++j) {
          auto& iterArgNode =
              std::get<VariableNode<SSAValue>>(node.iterArgs[j]->node_variant);
          iterArgNode.value = args[j];
        }

        assert(node.body != nullptr);
        assert(std::holds_alternative<YieldNode<SSAValue>>(
                   node.body->node_variant) &&
               "ForLoopNode body must be a YieldNode");
        std::vector<Value> bodyResults = processInternal(node.body);
        scf::YieldOp::create(builder, bodyResults);
      });

  std::vector<Value> results(loop.getResults().begin(),
                             loop.getResults().end());
  return results;
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const YieldNode<SSAValue>& node) {
  std::vector<Value> eltResults;
  eltResults.reserve(node.elements.size());
  for (const auto& elt : node.elements) {
    std::vector<Value> values = processInternal(elt);
    assert(values.size() == 1 && "Yield operands must be single values");
    eltResults.push_back(values[0]);
  }
  return eltResults;
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ResultAtNode<SSAValue>& node) {
  std::vector<Value> operands = processInternal(node.operand);
  return {operands[node.index]};
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
