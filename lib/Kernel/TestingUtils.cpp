#include "lib/Kernel/TestingUtils.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

namespace mlir {
namespace heir {
namespace kernel {

LiteralValue EvalVisitor::operator()(const LeafNode<LiteralValue>& node) {
  const auto& nodeVal = node.value.getTensor();
  const auto* vecVal = std::get_if<std::vector<int>>(&nodeVal);
  const auto* matVal = std::get_if<std::vector<std::vector<int>>>(&nodeVal);
  if (vecVal) {
    assert(vecVal->size() == node.value.getShape()[0]);
  }
  if (matVal) {
    assert(matVal->size() == node.value.getShape()[0]);
  }
  return node.value.getTensor();
}

LiteralValue EvalVisitor::operator()(const AddNode<LiteralValue>& node) {
  // Recursive calls use the public `process` method from the base class
  // to ensure caching is applied at every step.
  auto left = this->process(node.left);
  auto right = this->process(node.right);
  auto dim = left.getShape()[0];
  const auto& lVal = left.getTensor();
  const auto& rVal = right.getTensor();
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  assert(lVec && rVec && "unsupported add operands");
  assert(left.getShape() == right.getShape() && "disagreeing shapes");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] + (*rVec)[i];
  }
  return result;
}

LiteralValue EvalVisitor::operator()(const SubtractNode<LiteralValue>& node) {
  auto left = this->process(node.left);
  auto right = this->process(node.right);
  auto dim = left.getShape()[0];
  const auto& lVal = left.getTensor();
  const auto& rVal = right.getTensor();
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  assert(lVec && rVec && "unsupported sub operands");
  assert(left.getShape() == right.getShape() && "disagreeing shapes");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] - (*rVec)[i];
  }
  return result;
}

LiteralValue EvalVisitor::operator()(const MultiplyNode<LiteralValue>& node) {
  auto left = this->process(node.left);
  auto right = this->process(node.right);
  auto dim = left.getShape()[0];
  const auto& lVal = left.getTensor();
  const auto& rVal = right.getTensor();
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  assert(lVec && rVec && "unsupported mul operands");
  assert(left.getShape() == right.getShape() && "disagreeing shapes");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] * (*rVec)[i];
  }
  return result;
}

// Cyclic left-rotation by a given index
LiteralValue EvalVisitor::operator()(const LeftRotateNode<LiteralValue>& node) {
  auto operand = this->process(node.operand);
  auto dim = operand.getShape()[0];
  int amount = node.shift;
  const auto& oVal = operand.getTensor();
  const auto* oVec = std::get_if<std::vector<int>>(&oVal);
  assert(oVec && "unsupported rotate operand");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*oVec)[(i + amount) % oVec->size()];
  }
  return result;
}

LiteralValue EvalVisitor::operator()(const ExtractNode<LiteralValue>& node) {
  auto tensor = this->process(node.operand);
  unsigned index = node.index;
  return std::visit(
      [&](auto&& t) -> LiteralValue {
        // We can only extract from a 2D vector.
        if constexpr (std::is_same_v<std::decay_t<decltype(t)>,
                                     std::vector<std::vector<int>>>) {
          return t[index];
        }
        assert(false && "Unsupported type for extraction");
        return LiteralValue(std::vector<int>({}));
      },
      tensor.getTensor());
}

LiteralValue evalKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  EvalVisitor visitor;
  return visitor.process(dag);
}

std::string PrintVisitor::operator()(const LeafNode<LiteralValue>& node) {
  const auto& nodeVal = node.value.getTensor();
  const auto* vecVal = std::get_if<std::vector<int>>(&nodeVal);
  const auto* matVal = std::get_if<std::vector<std::vector<int>>>(&nodeVal);
  if (vecVal) {
    assert(vecVal->size() == node.value.getShape()[0]);
  }
  if (matVal) {
    assert(matVal->size() == node.value.getShape()[0]);
  }

  // just give a name to the vec
  if (vecVal) {
    return "v";
  }

  if (matVal) {
    return "Mat(...)";
  }

  return "UnknownLeaf";
}

std::string PrintVisitor::operator()(const AddNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
  return "(" + left + " + " + right + ")";
}

std::string PrintVisitor::operator()(const SubtractNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
  return "(" + left + " - " + right + ")";
}

std::string PrintVisitor::operator()(const MultiplyNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
  return "(" + left + " * " + right + ")";
}

std::string PrintVisitor::operator()(const LeftRotateNode<LiteralValue>& node) {
  std::string operand = this->process(node.operand);
  return "Rot(" + operand + ", " + std::to_string(node.shift) + ")";
}

std::string PrintVisitor::operator()(const ExtractNode<LiteralValue>& node) {
  // In these tests, extracting will always be from a plaintext matrix,
  // and the textual form of the entire matrix is too verbose. Could also
  // run a simplification on the generated kernel to inline the extracted
  // tensor instead of printing recursively.
  return "pt(" + std::to_string(node.index) + ")";
}

std::string printKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  PrintVisitor visitor;
  return visitor.process(dag);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
