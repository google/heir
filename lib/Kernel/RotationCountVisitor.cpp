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

int64_t RotationCountVisitor::operator()(
    const VariableNode<SymbolicValue>& node) {
  // Variables are typically placeholders - treat as plaintext by default
  nodeSecretStatus[currentNode] = false;
  return 0;
}

int64_t RotationCountVisitor::operator()(
    const ForLoopNode<SymbolicValue>& node) {
  const auto* thisNode = currentNode;  // Save before recursion

  // Process the init value
  int64_t initCount = processInternal(node.init);

  // Process the induction variable and iter_arg (these are variables)
  int64_t inductionVarCount = processInternal(node.inductionVar);
  int64_t iterArgCount = processInternal(node.iterArg);

  // Process the body if it exists
  int64_t bodyCount = 0;
  if (node.body) {
    bodyCount = processInternal(node.body);
  }

  // A loop's secret status is determined by its init and body
  bool initIsSecret = nodeSecretStatus[node.init.get()];
  bool bodyIsSecret = node.body ? nodeSecretStatus[node.body.get()] : false;
  nodeSecretStatus[thisNode] = initIsSecret || bodyIsSecret;

  // Total rotations are from init + body rotations * number of iterations
  // For now, we'll count the body rotations only once (conservative estimate)
  return initCount + inductionVarCount + iterArgCount + bodyCount;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
