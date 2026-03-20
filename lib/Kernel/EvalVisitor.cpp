#include "lib/Kernel/EvalVisitor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/Support/Debug.h"     // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

#define DEBUG_TYPE "eval-visitor"

namespace mlir {
namespace heir {
namespace kernel {

EvalResults EvalVisitor::operator()(const LeafNode<LiteralValue>& node) {
  LDBG() << "Visiting LeafNode";
  const auto& nodeVal = node.value.get();
  const auto* vecVal = std::get_if<std::vector<int>>(&nodeVal);
  const auto* matVal = std::get_if<std::vector<std::vector<int>>>(&nodeVal);
  if (vecVal) {
    assert(vecVal->size() == node.value.getShape()[0]);
  }
  if (matVal) {
    assert(matVal->size() == node.value.getShape()[0]);
  }

  return {node.value.get()};
}

EvalResults EvalVisitor::operator()(const AddNode<LiteralValue>& node) {
  LDBG() << "Visiting AddNode";

  // Recursive calls use the public `process` method from the base class
  // to ensure caching is applied at every step.
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];
  const auto& lVal = left.get();
  const auto& rVal = right.get();

  // Handle scalar addition
  const auto* lScalar = std::get_if<int>(&lVal);
  const auto* rScalar = std::get_if<int>(&rVal);
  if (lScalar && rScalar) {
    return {LiteralValue(*lScalar + *rScalar)};
  }

  // Handle vector addition
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  if (lVec && rVec) {
    assert(left.getShape() == right.getShape() && "disagreeing shapes");
    std::vector<int> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*lVec)[i] + (*rVec)[i];
    }
    return {result};
  }

  // If types are not supported, just return left as dummy
  return {left};
}

EvalResults EvalVisitor::operator()(const SubtractNode<LiteralValue>& node) {
  LDBG() << "Visiting SubtractNode";
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];
  const auto& lVal = left.get();
  const auto& rVal = right.get();

  // Handle scalar subtraction
  const auto* lScalar = std::get_if<int>(&lVal);
  const auto* rScalar = std::get_if<int>(&rVal);
  if (lScalar && rScalar) {
    return {LiteralValue(*lScalar - *rScalar)};
  }

  // Handle vector subtraction
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  if (lVec && rVec) {
    assert(left.getShape() == right.getShape() && "disagreeing shapes");
    std::vector<int> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*lVec)[i] - (*rVec)[i];
    }
    return {result};
  }

  return {left};
}

EvalResults EvalVisitor::operator()(const MultiplyNode<LiteralValue>& node) {
  LDBG() << "Visiting MultiplyNode";
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];
  const auto& lVal = left.get();
  const auto& rVal = right.get();

  // Handle scalar multiplication
  const auto* lScalar = std::get_if<int>(&lVal);
  const auto* rScalar = std::get_if<int>(&rVal);
  if (lScalar && rScalar) {
    return {LiteralValue(*lScalar * *rScalar)};
  }

  // Handle vector multiplication
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  if (lVec && rVec) {
    assert(left.getShape() == right.getShape() && "disagreeing shapes");
    std::vector<int> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*lVec)[i] * (*rVec)[i];
    }
    return {result};
  }
  return {left};
}

EvalResults EvalVisitor::operator()(const FloorDivNode<LiteralValue>& node) {
  LDBG() << "Visiting FloorDivNode";
  auto left = this->process(node.left)[0];
  const auto& lVal = left.get();

  // Scalar case
  const auto* lScalar = std::get_if<int>(&lVal);
  if (*lScalar) {
    return {LiteralValue(*lScalar / node.divisor)};
  }

  // Vector case
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  assert(lVec && "unsupported floorDiv operands");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] / node.divisor;
  }
  return {result};
}

// Cyclic left-rotation by a given index
EvalResults EvalVisitor::operator()(const LeftRotateNode<LiteralValue>& node) {
  LDBG() << "Visiting LeftRotateNode";
  auto operand = this->process(node.operand)[0];
  auto dim = operand.getShape().back();
  auto evaluatedShift = this->process(node.shift)[0];
  int amount = std::get<int>(evaluatedShift.get());
  // Normalize amount to be in [0, dim)
  amount = ((amount % dim) + dim) % dim;

  const auto& oVal = operand.get();
  if (const auto* oVec = std::get_if<std::vector<int>>(&oVal)) {
    std::vector<int> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*oVec)[(i + amount) % oVec->size()];
    }
    return {result};
  }

  // If the operand is not a 1D vector (e.g., a 2D float tensor), return as-is.
  return {operand};
}

EvalResults EvalVisitor::operator()(const ExtractNode<LiteralValue>& node) {
  LDBG() << "Visiting ExtractNode";
  auto tensor = this->process(node.operand)[0];

  // Evaluate the index expression to get an integer
  auto evaluatedIndex = this->process(node.index)[0];
  int index = std::get<int>(evaluatedIndex.get());

  return std::visit(
      [&](auto&& t) -> EvalResults {
        if constexpr (std::is_same_v<std::decay_t<decltype(t)>,
                                     std::vector<std::vector<int>>>) {
          return {LiteralValue(t[index])};
        } else if constexpr (std::is_same_v<std::decay_t<decltype(t)>,
                                            std::vector<int>>) {
          return {LiteralValue(t[index])};
        }
        assert(false && "Unsupported type for extraction");
        return {};
      },
      tensor.get());
}

EvalResults EvalVisitor::operator()(const ComparisonNode<LiteralValue>& node) {
  LDBG() << "Visiting ComparisonNode";
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];

  int lVal = std::get<int>(left.get());
  int rVal = std::get<int>(right.get());

  bool result = false;
  switch (node.predicate) {
    case ComparisonPredicate::LT:
      result = lVal < rVal;
      break;
    case ComparisonPredicate::LE:
      result = lVal <= rVal;
      break;
    case ComparisonPredicate::GT:
      result = lVal > rVal;
      break;
    case ComparisonPredicate::GE:
      result = lVal >= rVal;
      break;
    case ComparisonPredicate::EQ:
      result = lVal == rVal;
      break;
    case ComparisonPredicate::NE:
      result = lVal != rVal;
      break;
  }

  return {LiteralValue(result ? 1 : 0)};
}

EvalResults EvalVisitor::operator()(const IfElseNode<LiteralValue>& node) {
  LDBG() << "Visiting IfElseNode";
  auto condition = this->process(node.condition)[0];
  int condVal = std::get<int>(condition.get());

  if (condVal != 0) {
    return this->process(node.thenBody);
  }
  return this->process(node.elseBody);
}

EvalResults EvalVisitor::operator()(const ConstantTensorNode& node) {
  LDBG() << "Visiting ConstantTensorNode";
  // A bit of a hack, only support ints in testing
  std::vector<int> vec;
  vec.reserve(node.value.size());
  for (double v : node.value) {
    vec.push_back(static_cast<int>(v));
  }
  return {LiteralValue(vec)};
}

EvalResults EvalVisitor::operator()(const ConstantScalarNode& node) {
  LDBG() << "Visiting ConstantScalarNode";
  // A bit of a hack, casting the double to an int
  return {LiteralValue(static_cast<int>(node.value))};
}

EvalResults EvalVisitor::operator()(const SplatNode& node) {
  LDBG() << "Visiting SplatNode";
  // A bit of a hack, casting the double to an int
  int splatValue = static_cast<int>(node.value);

  // Check if this is a tensor type
  if (std::holds_alternative<kernel::IntTensorType>(node.type.type_variant)) {
    const auto& tensorType =
        std::get<kernel::IntTensorType>(node.type.type_variant);
    // Compute total size as product of all dimensions
    int64_t totalSize = 1;
    for (int64_t dim : tensorType.shape) {
      totalSize *= dim;
    }
    std::vector<int> result(totalSize, splatValue);
    return {LiteralValue(result)};
  } else if (std::holds_alternative<kernel::FloatTensorType>(
                 node.type.type_variant)) {
    const auto& tensorType =
        std::get<kernel::FloatTensorType>(node.type.type_variant);
    // Compute total size as product of all dimensions
    int64_t totalSize = 1;
    for (int64_t dim : tensorType.shape) {
      totalSize *= dim;
    }
    std::vector<int> result(totalSize, splatValue);
    return {LiteralValue(result)};
  }

  // Scalar type
  return {LiteralValue(splatValue)};
}

EvalResults EvalVisitor::operator()(const VariableNode<LiteralValue>& node) {
  assert(node.value.has_value() && "VariableNode value is not set");
  return {node.value.value()};
}

EvalResults EvalVisitor::operator()(const ForLoopNode<LiteralValue>& node) {
  LDBG() << "Visiting ForLoopNode";
  // Process initial values
  std::vector<LiteralValue> iterValues;
  iterValues.reserve(node.inits.size());
  for (const auto& init : node.inits) {
    EvalResults initResult = this->process(init);
    assert(initResult.size() == 1 && "Init must produce single value");
    iterValues.push_back(initResult[0]);
  }

  // Execute loop iterations
  for (int32_t i = node.lower; i < node.upper; i += node.step) {
    // Clear cache for the loop body since variables change each iteration
    this->clearSubtreeCache(node.body);

    // Set induction variable
    auto& inductionVarNode =
        std::get<VariableNode<LiteralValue>>(node.inductionVar->node_variant);
    inductionVarNode.value = LiteralValue(static_cast<int>(i));

    // Set iter args
    assert(node.iterArgs.size() == iterValues.size());
    for (size_t j = 0; j < node.iterArgs.size(); ++j) {
      auto& iterArgNode =
          std::get<VariableNode<LiteralValue>>(node.iterArgs[j]->node_variant);
      iterArgNode.value = iterValues[j];
    }

    // Execute body (should be a YieldNode)
    assert(node.body != nullptr);
    assert(std::holds_alternative<YieldNode<LiteralValue>>(
               node.body->node_variant) &&
           "ForLoopNode body must be a YieldNode");
    EvalResults bodyResults = this->process(node.body);

    // Update iter values for next iteration
    assert(bodyResults.size() == iterValues.size());
    iterValues = bodyResults;
  }

  return iterValues;
}

EvalResults EvalVisitor::operator()(const YieldNode<LiteralValue>& node) {
  LDBG() << "Visiting YieldNode";
  EvalResults results;
  results.reserve(node.elements.size());
  for (const auto& element : node.elements) {
    EvalResults elementResults = this->process(element);
    assert(elementResults.size() == 1 &&
           "Yield operands must be single values");
    results.push_back(elementResults[0]);
  }
  return results;
}

EvalResults EvalVisitor::operator()(const ResultAtNode<LiteralValue>& node) {
  LDBG() << "Visiting ResultAtNode";
  EvalResults operandResults = this->process(node.operand);
  assert(node.index < operandResults.size() && "Index out of bounds");
  return {operandResults[node.index]};
}

EvalResults evalKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  EvalVisitor visitor;
  return visitor.process(dag);
}

std::vector<EvalResults> multiEvalKernel(
    ArrayRef<std::shared_ptr<ArithmeticDagNode<LiteralValue>>> dags) {
  EvalVisitor visitor;
  return visitor.process(dags);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
