#include "lib/Kernel/IRMaterializingVisitor.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/Sequence.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

std::vector<Value> IRMaterializingVisitor::operator()(
    const LeafNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  return {node.value.getValue()};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ConstantScalarNode& node, ImplicitLocOpBuilder& builder) {
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
    const SplatNode& node, ImplicitLocOpBuilder& builder) {
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
    const EmptyNode& node, ImplicitLocOpBuilder& builder) {
  Type targetType = dagTypeToMLIRType(node.type, builder);
  RankedTensorType tensorTy = dyn_cast<RankedTensorType>(targetType);
  assert(tensorTy && "EmptyNode targetType must be a RankedTensorType");

  auto resType = RankedTensorType::get(node.shape, tensorTy.getElementType());
  auto emptyOp = tensor::EmptyOp::create(builder, resType, ValueRange{});
  createdOpCallback(emptyOp);
  return {emptyOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ConstantTensorNode& node, ImplicitLocOpBuilder& builder) {
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
    const AddNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  return binop<AddNode<SSAValue>, arith::AddFOp, arith::AddIOp>(node, builder);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const SubtractNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  return binop<SubtractNode<SSAValue>, arith::SubFOp, arith::SubIOp>(node,
                                                                     builder);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const MultiplyNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  return binop<MultiplyNode<SSAValue>, arith::MulFOp, arith::MulIOp>(node,
                                                                     builder);
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const FloorDivNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value lhs = this->process(node.left, builder)[0];
  // The assumption here is that FloorDiv is only used in kernels for scalar
  // index calculations, so the lhs.getType() should always be an integer or
  // index type.
  assert((isa<mlir::IntegerType>(lhs.getType()) ||
          isa<mlir::IndexType>(lhs.getType())) &&
         "FloorDiv lhs should be a scalar integer type");
  IntegerAttr rhsAttr = builder.getIntegerAttr(lhs.getType(), node.divisor);
  Value rhs = arith::ConstantOp::create(builder, rhsAttr);
  auto op = arith::DivSIOp::create(builder, lhs, rhs);
  createdOpCallback(op);
  return {op->getResult(0)};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const LeftRotateNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value operand = this->process(node.operand, builder)[0];
  Value shift = this->process(node.shift, builder)[0];
  auto rotateOp =
      tensor_ext::RotateOp::create(builder, operand.getType(), operand, shift);
  createdOpCallback(rotateOp);
  return {rotateOp};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ExtractNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value operand = this->process(node.operand, builder)[0];
  Value index = this->process(node.index, builder)[0];
  // Ensure index has index type
  if (!index.getType().isIndex()) {
    index = arith::IndexCastOp::create(builder, builder.getIndexType(), index);
  }

  RankedTensorType tensorType = cast<RankedTensorType>(operand.getType());

  if (tensorType.getRank() == 1) {
    auto extractOp =
        tensor::ExtractOp::create(builder, operand, ValueRange{index});
    createdOpCallback(extractOp);
    return {extractOp.getResult()};
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

  if (auto tensorTy = dyn_cast<RankedTensorType>(evaluatedType)) {
    // The extracted row keeps the operand's element type, which may differ
    // from the evaluated type's (e.g., an integer input of a float kernel).
    auto resultTy =
        RankedTensorType::get(tensorTy.getShape(), tensorType.getElementType());
    auto extractOp = tensor::ExtractSliceOp::create(builder, resultTy, operand,
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
    const InsertNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value scalar = this->process(node.scalar, builder)[0];
  Value dest = this->process(node.dest, builder)[0];
  Value index = this->process(node.index, builder)[0];
  // Ensure index has index type
  if (!index.getType().isIndex()) {
    index = arith::IndexCastOp::create(builder, builder.getIndexType(), index);
  }

  // This is like inserting a shape {N} tensor into an index of a {k, N} tensor.
  if (auto destTy = dyn_cast<RankedTensorType>(dest.getType())) {
    if (isa<RankedTensorType>(scalar.getType())) {
      SmallVector<OpFoldResult> offsets(destTy.getRank(),
                                        builder.getIndexAttr(0));
      offsets[0] = index;

      SmallVector<OpFoldResult> sizes;
      sizes.push_back(builder.getIndexAttr(1));
      for (int i = 1; i < destTy.getRank(); ++i) {
        sizes.push_back(builder.getIndexAttr(destTy.getDimSize(i)));
      }

      SmallVector<OpFoldResult> strides(destTy.getRank(),
                                        builder.getIndexAttr(1));
      auto op = tensor::InsertSliceOp::create(builder, scalar, dest, offsets,
                                              sizes, strides);
      createdOpCallback(op);
      return {op.getResult()};
    }
  }

  auto op = tensor::InsertOp::create(builder, scalar, dest, ValueRange{index});
  createdOpCallback(op);
  return {op.getResult()};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ComparisonNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value lhs = this->process(node.left, builder)[0];
  Value rhs = this->process(node.right, builder)[0];

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
    const IfElseNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value condition = this->process(node.condition, builder)[0];

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

  auto ifOp = scf::IfOp::create(
      builder, condition,
      [&](OpBuilder& nestedBuilder, Location nestedLoc) {
        ImplicitLocOpBuilder nestedB(nestedLoc, nestedBuilder);
        auto results = this->process(node.thenBody, nestedB);
        scf::YieldOp::create(nestedBuilder, nestedLoc, results);
      },
      [&](OpBuilder& nestedBuilder, Location nestedLoc) {
        ImplicitLocOpBuilder nestedB(nestedLoc, nestedBuilder);
        auto results = this->process(node.elseBody, nestedB);
        scf::YieldOp::create(nestedBuilder, nestedLoc, results);
      });

  createdOpCallback(ifOp);
  return std::vector<Value>(ifOp.getResults().begin(), ifOp.getResults().end());
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const VariableNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  assert(node.value.has_value() && "VariableNode value is not set");
  return {node.value->getValue()};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ForLoopNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  std::vector<Value> initValues;
  initValues.reserve(node.inits.size());
  for (const auto& init : node.inits) {
    Value initProcessed = this->process(init, builder)[0];
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

        ImplicitLocOpBuilder nestedB(nestedLoc, nestedBuilder);
        std::vector<Value> bodyResults = this->process(node.body, nestedB);
        scf::YieldOp::create(nestedB, bodyResults);
      });

  std::vector<Value> results(loop.getResults().begin(),
                             loop.getResults().end());
  return results;
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const YieldNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  std::vector<Value> eltResults;
  eltResults.reserve(node.elements.size());
  for (const auto& elt : node.elements) {
    std::vector<Value> values = this->process(elt, builder);
    assert(values.size() == 1 && "Yield operands must be single values");
    eltResults.push_back(values[0]);
  }
  return eltResults;
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ResultAtNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  std::vector<Value> operands = this->process(node.operand, builder);
  return {operands[node.index]};
}

std::vector<Value> IRMaterializingVisitor::operator()(
    const ElementwiseOpNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  SmallVector<Value> operands;
  for (const auto& operand : node.operands) {
    operands.push_back(this->process(operand, builder)[0]);
  }
  auto shapedType = cast<RankedTensorType>(operands.front().getType());
  SmallVector<Type> resultTypes;
  for (Type type : node.op->getResultTypes()) {
    resultTypes.push_back(shapedType.clone(type));
  }
  OperationState state(builder.getLoc(), node.op->getName(), operands,
                       resultTypes, node.op->getAttrs());
  Operation* op = builder.create(state);
  createdOpCallback(op);
  return {op->getResults().begin(), op->getResults().end()};
}

// Constant and scalar sources become a constant or a splat; any other source
// is read by a loop over constant index tables.
std::vector<Value> IRMaterializingVisitor::operator()(
    const GatherNode<SSAValue>& node, ImplicitLocOpBuilder& builder) {
  Value source = node.source ? this->process(node.source, builder)[0] : Value();
  Type elementType =
      source ? getElementTypeOrSelf(source.getType()) : builder.getIndexType();
  auto resultType = RankedTensorType::get(node.shape, elementType);
  Attribute zero = builder.getZeroAttr(elementType);
  auto create = [&](auto op) {
    createdOpCallback(op);
    return op;
  };

  Attribute constant;
  if (!source || matchPattern(source, m_Constant(&constant))) {
    auto elements = dyn_cast_if_present<ElementsAttr>(constant);
    std::function<Attribute(int64_t)> read;
    if (elements) read = constantElementReader(elements);
    if (!elements || read) {
      SmallVector<Attribute> result;
      for (int64_t index : node.indices) {
        if (index < 0) {
          result.push_back(zero);
        } else if (!source) {
          result.push_back(builder.getIndexAttr(index));
        } else {
          result.push_back(read ? read(index) : constant);
        }
      }
      return {create(arith::ConstantOp::create(
          builder, resultType, DenseElementsAttr::get(resultType, result)))};
    }
  }

  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType &&
      llvm::all_of(node.indices, [](int64_t index) { return index >= 0; })) {
    return {create(tensor::SplatOp::create(builder, resultType, source))};
  }
  Value flat = source;
  if (!sourceType) {
    flat = create(tensor::FromElementsOp::create(builder, source));
  } else if (sourceType.getRank() == 0) {
    flat = create(tensor::ExpandShapeOp::create(
        builder, RankedTensorType::get({1}, elementType), source,
        ArrayRef<ReassociationIndices>{}));
  } else if (sourceType.getRank() > 1) {
    flat = create(tensor::CollapseShapeOp::create(
        builder, source,
        ArrayRef<ReassociationIndices>{
            llvm::to_vector(llvm::seq<int64_t>(sourceType.getRank()))}));
  }
  SmallVector<int64_t> from, to;
  for (auto [i, index] : llvm::enumerate(node.indices)) {
    if (index < 0) continue;
    from.push_back(index);
    to.push_back(i);
  }
  auto table = [&](ArrayRef<int64_t> values) -> Value {
    auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                      builder.getIndexType());
    return create(arith::ConstantOp::create(
        builder, type, DenseElementsAttr::get(type, values)));
  };
  Value fromTable = table(from), toTable = table(to);
  auto flatType =
      RankedTensorType::get({resultType.getNumElements()}, elementType);
  Value init = create(arith::ConstantOp::create(
      builder, flatType, DenseElementsAttr::get(flatType, zero)));
  Value lower = create(arith::ConstantIndexOp::create(builder, 0));
  Value upper = create(arith::ConstantIndexOp::create(builder, from.size()));
  Value step = create(arith::ConstantIndexOp::create(builder, 1));
  auto loop = scf::ForOp::create(
      builder, lower, upper, step, ValueRange{init},
      [&](OpBuilder& nested, Location loc, Value k, ValueRange iterArgs) {
        Value src = tensor::ExtractOp::create(nested, loc, fromTable, k);
        Value dst = tensor::ExtractOp::create(nested, loc, toTable, k);
        Value element = tensor::ExtractOp::create(nested, loc, flat, src);
        Value updated =
            tensor::InsertOp::create(nested, loc, element, iterArgs[0], dst);
        scf::YieldOp::create(nested, loc, updated);
      });
  loop->walk([&](Operation* op) { createdOpCallback(op); });
  SmallVector<ReassociationIndices> reassociation = {
      llvm::to_vector(llvm::seq<int64_t>(resultType.getRank()))};
  return {create(tensor::ExpandShapeOp::create(
      builder, resultType, loop.getResult(0), reassociation))};
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
