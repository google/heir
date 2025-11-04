#include "lib/Kernel/MultiplicativeDepthVisitor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <variant>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

int64_t MultiplicativeDepthVisitor::process(
    const std::shared_ptr<NodeTy>& node) {
  nodeDepth_.clear();
  visitedNodes_.clear();
  return processInternal(node);
}

int64_t MultiplicativeDepthVisitor::processInternal(
    const std::shared_ptr<NodeTy>& node) {
  const auto* nodePtr = node.get();

  // Check if we've already computed depth for this node (memoization)
  auto it = nodeDepth_.find(nodePtr);
  if (it != nodeDepth_.end()) {
    return it->second;
  }

  // Cycle detection
  if (visitedNodes_.count(nodePtr)) {
    return 0;  // Shouldn't happen in a DAG
  }
  visitedNodes_.insert(nodePtr);

  // Set current node before dispatching to visitor methods
  currentNode_ = nodePtr;

  // Use type dispatching via std::visit
  int64_t depth = std::visit(*this, node->node_variant);

  // Cache the result
  nodeDepth_[nodePtr] = depth;
  visitedNodes_.erase(nodePtr);

  return depth;
}

// Type-dispatched visitor methods

int64_t MultiplicativeDepthVisitor::operator()(
    const ConstantScalarNode& node) {
  // Constants have depth 0
  return 0;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const ConstantTensorNode& node) {
  // Constants have depth 0
  return 0;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const LeafNode<SymbolicValue>& node) {
  // Leaf nodes (inputs) have depth 0
  return 0;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const AddNode<SymbolicValue>& node) {
  // Addition doesn't increase depth
  // Depth is max of operands
  int64_t leftDepth = processInternal(node.left);
  int64_t rightDepth = processInternal(node.right);

  return std::max(leftDepth, rightDepth);
}

int64_t MultiplicativeDepthVisitor::operator()(
    const SubtractNode<SymbolicValue>& node) {
  // Subtraction doesn't increase depth
  // Depth is max of operands
  int64_t leftDepth = processInternal(node.left);
  int64_t rightDepth = processInternal(node.right);

  return std::max(leftDepth, rightDepth);
}

int64_t MultiplicativeDepthVisitor::operator()(
    const MultiplyNode<SymbolicValue>& node) {
  // Multiplication increases depth by 1
  // Total depth is max of operand depths + 1
  int64_t leftDepth = processInternal(node.left);
  int64_t rightDepth = processInternal(node.right);

  return std::max(leftDepth, rightDepth) + 1;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const PowerNode<SymbolicValue>& node) {
  // Power operation: x^n requires ceil(log2(n)) multiplications
  // Using repeated squaring: x^8 = ((x^2)^2)^2 requires 3 multiplications
  int64_t baseDepth = processInternal(node.base);

  if (node.exponent <= 1) {
    // x^0 = 1, x^1 = x, no additional depth
    return baseDepth;
  }

  // Additional depth from exponentiation
  int64_t expDepth = static_cast<int64_t>(std::ceil(std::log2(node.exponent)));

  return baseDepth + expDepth;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const LeftRotateNode<SymbolicValue>& node) {
  // Rotation doesn't increase depth
  // It's a permutation operation (automorphism in FHE)
  return processInternal(node.operand);
}

int64_t MultiplicativeDepthVisitor::operator()(
    const ExtractNode<SymbolicValue>& node) {
  // Extract doesn't increase depth
  // It's just selecting elements from a tensor
  return processInternal(node.operand);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
