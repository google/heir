#ifndef LIB_KERNEL_ROTATIONCOUNTVISITOR_H_
#define LIB_KERNEL_ROTATIONCOUNTVISITOR_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

// Visitor to count unique rotations in a symbolic DAG with CSE deduplication.
// This visitor traverses an ArithmeticDAG and counts rotation operations,
// ensuring that shared subexpressions (common subexpression elimination) are
// only counted once.
class RotationCountVisitor {
 public:
  using NodeTy = ArithmeticDagNode<SymbolicValue>;

  RotationCountVisitor() {}

  // Main entry point - counts rotations in the DAG
  int64_t process(const std::shared_ptr<NodeTy>& node);

  // Type-dispatched visit methods
  int64_t operator()(const ConstantScalarNode& node);
  int64_t operator()(const ConstantTensorNode& node);
  int64_t operator()(const LeafNode<SymbolicValue>& node);
  int64_t operator()(const AddNode<SymbolicValue>& node);
  int64_t operator()(const SubtractNode<SymbolicValue>& node);
  int64_t operator()(const MultiplyNode<SymbolicValue>& node);
  int64_t operator()(const PowerNode<SymbolicValue>& node);
  int64_t operator()(const LeftRotateNode<SymbolicValue>& node);
  int64_t operator()(const ExtractNode<SymbolicValue>& node);

 private:
  std::unordered_set<const NodeTy*> visitedNodes;
  std::unordered_map<const NodeTy*, bool> nodeSecretStatus;
  const NodeTy* currentNode = nullptr;

  // Internal recursive traversal with CSE tracking
  int64_t processInternal(const std::shared_ptr<NodeTy>& node);
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_ROTATIONCOUNTVISITOR_H_
