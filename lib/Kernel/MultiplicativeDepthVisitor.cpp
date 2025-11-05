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
  nodeSecretStatus_.clear();
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
  // Constants are always plaintext
  nodeSecretStatus_[currentNode_] = false;
  return 0;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const ConstantTensorNode& node) {
  // Constants are always plaintext
  nodeSecretStatus_[currentNode_] = false;
  return 0;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const LeafNode<SymbolicValue>& node) {
  // Leaf nodes get their secret status from the SymbolicValue
  nodeSecretStatus_[currentNode_] = node.value.isSecret();
  return 0;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const AddNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode_;  // Save before recursion

  // Addition doesn't increase depth
  // Depth is max of operands
  int64_t leftDepth = processInternal(node.left);
  int64_t rightDepth = processInternal(node.right);

  // A value is secret if any operand is secret
  bool leftIsSecret = nodeSecretStatus_[node.left.get()];
  bool rightIsSecret = nodeSecretStatus_[node.right.get()];
  nodeSecretStatus_[thisNode] = leftIsSecret || rightIsSecret;

  return std::max(leftDepth, rightDepth);
}

int64_t MultiplicativeDepthVisitor::operator()(
    const SubtractNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode_;  // Save before recursion

  // Subtraction doesn't increase depth
  // Depth is max of operands
  int64_t leftDepth = processInternal(node.left);
  int64_t rightDepth = processInternal(node.right);

  // A value is secret if any operand is secret
  bool leftIsSecret = nodeSecretStatus_[node.left.get()];
  bool rightIsSecret = nodeSecretStatus_[node.right.get()];
  nodeSecretStatus_[thisNode] = leftIsSecret || rightIsSecret;

  return std::max(leftDepth, rightDepth);
}

int64_t MultiplicativeDepthVisitor::operator()(
    const MultiplyNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode_;  // Save before recursion

  int64_t leftDepth = processInternal(node.left);
  int64_t rightDepth = processInternal(node.right);

  // Check operand secret status
  bool leftIsSecret = nodeSecretStatus_[node.left.get()];
  bool rightIsSecret = nodeSecretStatus_[node.right.get()];

  // Result is secret if any operand is secret
  nodeSecretStatus_[thisNode] = leftIsSecret || rightIsSecret;

  // CRITICAL: Only ciphertext × ciphertext increases depth!
  // Plaintext × ciphertext is depth 0 (free scalar multiplication in FHE)
  if (leftIsSecret && rightIsSecret) {
    // Both operands are ciphertext: expensive, increases depth
    return std::max(leftDepth, rightDepth) + 1;
  } else {
    // At least one operand is plaintext: free operation
    return std::max(leftDepth, rightDepth);
  }
}

int64_t MultiplicativeDepthVisitor::operator()(
    const PowerNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode_;  // Save before recursion

  int64_t baseDepth = processInternal(node.base);

  // Secret status is inherited from the base
  bool baseIsSecret = nodeSecretStatus_[node.base.get()];
  nodeSecretStatus_[thisNode] = baseIsSecret;

  if (node.exponent <= 1) {
    // x^0 = 1, x^1 = x, no additional depth
    return baseDepth;
  }

  // Power operation: x^n requires ceil(log2(n)) multiplications
  // Using repeated squaring: x^8 = ((x^2)^2)^2 requires 3 multiplications
  // But only if x is ciphertext! If x is plaintext, power is free.
  if (baseIsSecret) {
    // Ciphertext power: expensive
    int64_t expDepth = static_cast<int64_t>(std::ceil(std::log2(node.exponent)));
    return baseDepth + expDepth;
  } else {
    // Plaintext power: free
    return baseDepth;
  }
}

int64_t MultiplicativeDepthVisitor::operator()(
    const LeftRotateNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode_;  // Save before recursion

  // Rotation doesn't increase depth
  // It's a permutation operation (automorphism in FHE)
  int64_t operandDepth = processInternal(node.operand);

  // Secret status is inherited from operand
  bool operandIsSecret = nodeSecretStatus_[node.operand.get()];
  nodeSecretStatus_[thisNode] = operandIsSecret;

  return operandDepth;
}

int64_t MultiplicativeDepthVisitor::operator()(
    const ExtractNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode_;  // Save before recursion

  // Extract doesn't increase depth
  // It's just selecting elements from a tensor
  int64_t operandDepth = processInternal(node.operand);

  // Secret status is inherited from operand
  bool operandIsSecret = nodeSecretStatus_[node.operand.get()];
  nodeSecretStatus_[thisNode] = operandIsSecret;

  return operandDepth;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
