#include "lib/Kernel/RotationCountVisitor.h"

#include <cstdint>
#include <memory>
#include <variant>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

int64_t RotationCountVisitor::process(const std::shared_ptr<NodeTy>& node) {
  visitedNodes.clear();
  nodeSecretStatus.clear();
  return processInternal(node);
}

int64_t RotationCountVisitor::processInternal(
    const std::shared_ptr<NodeTy>& node) {
  const auto* nodePtr = node.get();

  // If we've already visited this node, don't count it again (CSE)
  if (visitedNodes.count(nodePtr)) {
    return 0;
  }
  visitedNodes.insert(nodePtr);

  // Set current node before dispatching to visitor methods
  currentNode = nodePtr;

  // Use type dispatching via std::visit
  return std::visit(*this, node->node_variant);
}

// Type-dispatched visitor methods

int64_t RotationCountVisitor::operator()(const ConstantScalarNode& node) {
  // Constants are always plaintext
  nodeSecretStatus[currentNode] = false;
  return 0;
}

int64_t RotationCountVisitor::operator()(const ConstantTensorNode& node) {
  // Constants are always plaintext
  nodeSecretStatus[currentNode] = false;
  return 0;
}

int64_t RotationCountVisitor::operator()(const LeafNode<SymbolicValue>& node) {
  // Leaf nodes get their secret status from the SymbolicValue
  nodeSecretStatus[currentNode] = node.value.isSecret();
  return 0;
}

int64_t RotationCountVisitor::operator()(const AddNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  int64_t leftCount = processInternal(node.left);
  int64_t rightCount = processInternal(node.right);

  // A value is secret if any operand is secret
  bool leftIsSecret = nodeSecretStatus[node.left.get()];
  bool rightIsSecret = nodeSecretStatus[node.right.get()];
  nodeSecretStatus[thisNode] = leftIsSecret || rightIsSecret;

  return leftCount + rightCount;
}

int64_t RotationCountVisitor::operator()(
    const SubtractNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  int64_t leftCount = processInternal(node.left);
  int64_t rightCount = processInternal(node.right);

  // A value is secret if any operand is secret
  bool leftIsSecret = nodeSecretStatus[node.left.get()];
  bool rightIsSecret = nodeSecretStatus[node.right.get()];
  nodeSecretStatus[thisNode] = leftIsSecret || rightIsSecret;

  return leftCount + rightCount;
}

int64_t RotationCountVisitor::operator()(
    const MultiplyNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  int64_t leftCount = processInternal(node.left);
  int64_t rightCount = processInternal(node.right);

  // A value is secret if any operand is secret
  bool leftIsSecret = nodeSecretStatus[node.left.get()];
  bool rightIsSecret = nodeSecretStatus[node.right.get()];
  nodeSecretStatus[thisNode] = leftIsSecret || rightIsSecret;

  return leftCount + rightCount;
}

int64_t RotationCountVisitor::operator()(const PowerNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  // Power operations don't introduce rotations, just process the base
  int64_t baseCount = processInternal(node.base);

  // Secret status is inherited from the base
  bool baseIsSecret = nodeSecretStatus[node.base.get()];
  nodeSecretStatus[thisNode] = baseIsSecret;

  return baseCount;
}

int64_t RotationCountVisitor::operator()(
    const LeftRotateNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  int64_t operandCount = processInternal(node.operand);

  // Get operand's secret status
  bool operandIsSecret = nodeSecretStatus[node.operand.get()];

  // Secret status is inherited from operand
  nodeSecretStatus[thisNode] = operandIsSecret;

  // Only count rotation if operand is secret (encrypted)
  // Plaintext rotations are free!
  return operandCount + (operandIsSecret ? 1 : 0);
}

int64_t RotationCountVisitor::operator()(
    const ExtractNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  int64_t operandCount = processInternal(node.operand);

  // Secret status is inherited from operand
  bool operandIsSecret = nodeSecretStatus[node.operand.get()];
  nodeSecretStatus[thisNode] = operandIsSecret;

  return operandCount;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
