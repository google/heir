#include "lib/Analysis/RotationAnalysis/DagBuilder.h"

#include <cassert>
#include <memory>
#include <vector>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {

using kernel::ArithmeticDagNode;
using kernel::DagType;
using kernel::LiteralValue;
using kernel::mlirTypeToDagType;

using Node = ArithmeticDagNode<LiteralValue>;
using NodePtr = std::shared_ptr<Node>;

NodePtr DagBuilder::findNodeOrMakeNewVariable(Value value) {
  LDBG() << "Finding node for value " << value;
  if (valueToNode.contains(value)) {
    LDBG() << "Found in env";
    return valueToNode[value];
  }

  DagType dagType = mlirTypeToDagType(value.getType());
  IntegerAttr attr;
  if (matchPattern(value, m_Constant(&attr))) {
    LDBG() << "Matched to constant scalar " << attr;
    auto node = Node::constantScalar(attr.getInt(), dagType);
    valueToNode[value] = node;
    return node;
  }

  DenseIntElementsAttr elementsAttr;
  if (matchPattern(value, m_Constant(&elementsAttr))) {
    LDBG() << "Matched to constant tensor " << elementsAttr;
    std::vector<double> values;
    values.reserve(elementsAttr.size());
    for (APInt value : elementsAttr) {
      values.push_back(value.getSExtValue());
    }
    auto node = Node::constantTensor(values, dagType);
    valueToNode[value] = node;
    return node;
  }

  LDBG() << "Unmatched, treating as variable; value=" << value;
  // In this case, the value is either a block argument, an OpResult of an op
  // that hasn't been visited yet, or a value whose type does not materialize to
  // an index. We treat these nodes as variables for the purpose of rotation
  // analysis.
  auto res = Node::variable(dagType);
  valueToNode[value] = res;
  return res;
}

FailureOr<NodePtr> DagBuilder::visitBlockWithSingleTerminator(Block* block) {
  LDBG() << "Visiting block";
  NodePtr last;
  for (Operation& op : block->getOperations()) {
    auto res = build(&op);
    if (failed(res)) return failure();

    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      LDBG() << "Found terminator " << op;
      assert(!last && "expected single terminator");
      last = res.value();
    }
  }

  if (!last) return failure();
  return last;
}

FailureOr<NodePtr> DagBuilder::visit(scf::ForOp op) {
  IntegerAttr lb, ub, step;
  if (!matchPattern(op.getLowerBound(), m_Constant(&lb)) ||
      !matchPattern(op.getUpperBound(), m_Constant(&ub)) ||
      !matchPattern(op.getStep(), m_Constant(&step))) {
    LDBG() << "Loop bounds must be constant for RotationAnalysis";
    return failure();
  }

  SmallVector<NodePtr> inits;
  SmallVector<DagType> initTypes;
  inits.reserve(op.getInits().size());
  for (Value init : op.getInits()) {
    NodePtr var = findNodeOrMakeNewVariable(init);
    valueToNode[init] = var;
    inits.push_back(var);
    initTypes.push_back(mlirTypeToDagType(init.getType()));
  }

  auto dagNode = Node::loop(
      inits, initTypes, lb.getInt(), ub.getInt(), step.getInt(),
      [&](NodePtr inductionVar, const std::vector<NodePtr>& iterArgs) {
        // Set up valueToNode for the purpose of visiting the body.
        valueToNode[op.getInductionVar()] = inductionVar;
        for (const auto& [val, node] :
             llvm::zip(op.getRegionIterArgs(), iterArgs)) {
          valueToNode[val] = node;
        }

        FailureOr<NodePtr> bodyRes =
            visitBlockWithSingleTerminator(op.getBody());
        assert(succeeded(bodyRes) && "failed to parse body");
        return *bodyRes;
      });

  for (OpResult opResult : op->getOpResults()) {
    valueToNode[opResult] = Node::resultAt(dagNode, opResult.getResultNumber());
  }
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(scf::IfOp op) {
  if (op.getElseRegion().empty()) {
    op.emitOpError()
        << "skipped: If/Else without an else branch is not supported\n";
    return failure();
  }

  auto condition = findNodeOrMakeNewVariable(op.getCondition());
  auto thenRes = visitBlockWithSingleTerminator(&op.getThenRegion().front());
  if (failed(thenRes)) return failure();
  NodePtr thenBody = thenRes.value();

  auto elseRes = visitBlockWithSingleTerminator(&op.getElseRegion().front());
  if (failed(elseRes)) return failure();
  NodePtr elseBody = elseRes.value();

  auto dagNode = Node::ifElse(condition, thenBody, elseBody);
  for (OpResult opResult : op->getOpResults()) {
    valueToNode[opResult] = Node::resultAt(dagNode, opResult.getResultNumber());
  }
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(scf::YieldOp op) {
  std::vector<NodePtr> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    operands.push_back(findNodeOrMakeNewVariable(operand));
  }
  return Node::yield(operands);
}

FailureOr<NodePtr> DagBuilder::visit(RotationOpInterface op) {
  LDBG() << "Processing RotationOpInterface " << op;
  auto indices = op.getRotationIndices();
  auto toShiftNode = [&](OpFoldResult ofr) -> NodePtr {
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      auto intAttr = cast<IntegerAttr>(attr);
      return Node::constantScalar(intAttr.getInt(),
                                  mlirTypeToDagType(intAttr.getType()));
    }
    return findNodeOrMakeNewVariable(cast<Value>(ofr));
  };

  // Find the rotatable operand.
  OpOperand* rotatedOperand = op.getRotatedOperand();

  if (!rotatedOperand) {
    LDBG() << "Could not find a rotated operand for op " << op
           << ". It may be that the default implementation of "
              "getRotatedOperand is incorrect for this RotationOpInterface.";
    return failure();
  }

  auto tensorNode = findNodeOrMakeNewVariable(rotatedOperand->get());

  NodePtr dagNode;
  if (indices.size() == 1) {
    dagNode = Node::leftRotate(tensorNode, toShiftNode(indices[0]));
  } else {
    std::vector<NodePtr> shifts;
    shifts.reserve(indices.size());
    for (auto& ofr : indices) {
      shifts.push_back(toShiftNode(ofr));
    }
    dagNode = Node::leftRotateBulk(tensorNode, std::move(shifts));
  }
  valueToNode[op->getResult(0)] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::MulIOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());
  auto dagNode = Node::mul(lhs, rhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::AddIOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());
  auto dagNode = Node::add(lhs, rhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::SubIOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());
  auto dagNode = Node::sub(lhs, rhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::CmpIOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());

  kernel::ComparisonPredicate pred;
  switch (op.getPredicate()) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      pred = kernel::ComparisonPredicate::LT;
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      pred = kernel::ComparisonPredicate::LE;
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      pred = kernel::ComparisonPredicate::GT;
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      pred = kernel::ComparisonPredicate::GE;
      break;
    case arith::CmpIPredicate::eq:
      pred = kernel::ComparisonPredicate::EQ;
      break;
    case arith::CmpIPredicate::ne:
      pred = kernel::ComparisonPredicate::NE;
      break;
  }

  auto dagNode = Node::comparison(lhs, rhs, pred);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::ConstantOp op) {
  NodePtr dagNode =
      TypeSwitch<Attribute, NodePtr>(op.getValue())
          .Case<IntegerAttr>([&](auto attr) {
            return Node::constantScalar(
                attr.getInt(), mlirTypeToDagType(op.getResult().getType()));
          })
          .Case<DenseIntElementsAttr>([&](auto attr) {
            std::vector<double> vals;
            vals.reserve(attr.size());
            for (const APInt& val : attr) {
              vals.push_back(val.getSExtValue());
            }
            return Node::constantTensor(
                vals, mlirTypeToDagType(op.getResult().getType()));
          })
          .Default([&](auto type) { return nullptr; });

  if (!dagNode) return failure();
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(tensor::ExtractOp op) {
  if (op.getIndices().size() > 1) return failure();

  auto tensor = findNodeOrMakeNewVariable(op.getTensor());
  auto index = findNodeOrMakeNewVariable(op.getIndices()[0]);
  auto dagNode = Node::extract(tensor, index);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(tensor::SplatOp op) {
  IntegerAttr attr;
  if (matchPattern(op.getInput(), m_Constant(&attr))) {
    LDBG() << "Matched splatted value to constant scalar " << attr;
  }

  auto dagNode =
      Node::splat(attr.getInt(), mlirTypeToDagType(op.getResult().getType()));
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::DivSIOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());

  IntegerAttr attr;
  if (matchPattern(op.getRhs(), m_Constant(&attr))) {
    LDBG() << "Matched divSI RHS to constant scalar " << attr;
  }

  auto dagNode = Node::floorDiv(lhs, attr.getInt());
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::AddFOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());
  auto dagNode = Node::add(lhs, rhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::MulFOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());
  auto dagNode = Node::mul(lhs, rhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::SubFOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getLhs());
  auto rhs = findNodeOrMakeNewVariable(op.getRhs());
  auto dagNode = Node::sub(lhs, rhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(arith::NegFOp op) {
  auto lhs = findNodeOrMakeNewVariable(op.getOperand());
  // ArithmeticDag doesn't have NegNode, so we can use (0 - lhs)
  auto zero = Node::constantScalar(0.0, mlirTypeToDagType(op.getType()));
  auto dagNode = Node::sub(zero, lhs);
  valueToNode[op.getResult()] = dagNode;
  return dagNode;
}

FailureOr<NodePtr> DagBuilder::visit(tensor::ExtractSliceOp op) {
  // For rotation analysis, we can approximate extract_slice as just returning
  // the source tensor if we don't care about the specific values, or we can
  // try to be more precise. Since rotation analysis usually only cares about
  // the fact that *some* tensor is being rotated, returning the source is
  // enough to keep the DAG connected.
  auto source = findNodeOrMakeNewVariable(op.getSource());
  valueToNode[op.getResult()] = source;
  return source;
}

FailureOr<NodePtr> DagBuilder::build(Operation* op) {
  LDBG() << "Visiting op " << *op;
  return llvm::TypeSwitch<Operation*, FailureOr<NodePtr>>(op)
      .Case<arith::AddFOp, arith::AddIOp, arith::CmpIOp, arith::ConstantOp,
            arith::DivSIOp, arith::MulFOp, arith::MulIOp, arith::SubFOp,
            arith::SubIOp, arith::NegFOp, scf::ForOp, scf::IfOp, scf::YieldOp,
            tensor::ExtractOp, tensor::ExtractSliceOp, tensor::SplatOp,
            RotationOpInterface>([&](auto op) { return visit(op); })
      .Default([&](Operation* op) -> FailureOr<NodePtr> {
        LDBG() << "Unsupported op type " << op->getName() << ", skipping";
        return NodePtr(nullptr);
      });
}

}  // namespace heir
}  // namespace mlir
