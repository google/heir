#include "lib/Kernel/IRMaterializingVisitor.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
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
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

namespace {}  // namespace

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
  Value index = this->process(node.index)[0];
  // Ensure index has index type
  if (!index.getType().isIndex()) {
    index = arith::IndexCastOp::create(builder, builder.getIndexType(), index);
  }

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

std::vector<Value> IRMaterializingVisitor::operator()(
    const ComparisonNode<SSAValue>& node) {
  Value lhs = this->process(node.left)[0];
  Value rhs = this->process(node.right)[0];

  auto op = TypeSwitch<Type, Operation*>(getElementTypeOrSelf(lhs.getType()))
                .Case<mlir::FloatType>([&](auto ty) {
                  arith::CmpFPredicate pred;
                  switch (node.predicate) {
                    case ComparisonPredicate::LT:
                      pred = arith::CmpFPredicate::OLT;
                      break;
                    case ComparisonPredicate::LE:
                      pred = arith::CmpFPredicate::OLE;
                      break;
                    case ComparisonPredicate::GT:
                      pred = arith::CmpFPredicate::OGT;
                      break;
                    case ComparisonPredicate::GE:
                      pred = arith::CmpFPredicate::OGE;
                      break;
                    case ComparisonPredicate::EQ:
                      pred = arith::CmpFPredicate::OEQ;
                      break;
                    case ComparisonPredicate::NE:
                      pred = arith::CmpFPredicate::ONE;
                      break;
                  }
                  return arith::CmpFOp::create(builder, pred, lhs, rhs);
                })
                .Case<mlir::IntegerType, mlir::IndexType>([&](auto ty) {
                  arith::CmpIPredicate pred;
                  switch (node.predicate) {
                    case ComparisonPredicate::LT:
                      pred = arith::CmpIPredicate::slt;
                      break;
                    case ComparisonPredicate::LE:
                      pred = arith::CmpIPredicate::sle;
                      break;
                    case ComparisonPredicate::GT:
                      pred = arith::CmpIPredicate::sgt;
                      break;
                    case ComparisonPredicate::GE:
                      pred = arith::CmpIPredicate::sge;
                      break;
                    case ComparisonPredicate::EQ:
                      pred = arith::CmpIPredicate::eq;
                      break;
                    case ComparisonPredicate::NE:
                      pred = arith::CmpIPredicate::ne;
                      break;
                  }
                  return arith::CmpIOp::create(builder, pred, lhs, rhs);
                })
                .Default([&](Type) {
                  llvm_unreachable("Unsupported type for comparison operation");
                  return nullptr;
                });
  createdOpCallback(op);
  return {op->getResult(0)};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const IfElseNode<SSAValue>& node) {
  Value condition = this->process(node.condition)[0];

  auto* thenYield =
      std::get_if<YieldNode<SSAValue>>(&node.thenBody->node_variant);
  assert(thenYield && "IfElseNode thenBody must be a YieldNode");
  auto* elseYield =
      std::get_if<YieldNode<SSAValue>>(&node.elseBody->node_variant);
  assert(elseYield && "IfElseNode elseBody must be a YieldNode");
  if (thenYield->elements.size() != elseYield->elements.size()) {
    assert(false && "If branches must yield same number of elements");
    return {};
  }

  SmallVector<Type> resultTypes;
  std::vector<Value> thenResults = this->process(node.thenBody);
  for (Value v : thenResults) {
    resultTypes.push_back(v.getType());
  }

  auto ifOp = scf::IfOp::create(
      builder, condition,
      [&](OpBuilder& nestedBuilder, Location nestedLoc) {
        scf::YieldOp::create(builder, thenResults);
      },
      [&](OpBuilder& nestedBuilder, Location nestedLoc) {
        std::vector<Value> thenResults = this->process(node.thenBody);
        scf::YieldOp::create(builder, thenResults);
      });

  createdOpCallback(ifOp);
  return std::vector<Value>(ifOp.getResults().begin(), ifOp.getResults().end());
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
    Value initProcessed = this->process(init)[0];
    initValues.push_back(initProcessed);
  }

  Value lower = arith::ConstantIndexOp::create(builder, node.lower);
  Value upper = arith::ConstantIndexOp::create(builder, node.upper);
  Value step = arith::ConstantIndexOp::create(builder, node.step);
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
        std::vector<Value> bodyResults = this->process(node.body);
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
    std::vector<Value> values = this->process(elt);
    assert(values.size() == 1 && "Yield operands must be single values");
    eltResults.push_back(values[0]);
  }
  return eltResults;
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ResultAtNode<SSAValue>& node) {
  std::vector<Value> operands = this->process(node.operand);
  return {operands[node.index]};
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
