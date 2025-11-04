#ifndef LIB_KERNEL_MULTIPLICATIVEDEPTHVISITOR_H_
#define LIB_KERNEL_MULTIPLICATIVEDEPTHVISITOR_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

// Visitor to compute the multiplicative depth of a symbolic DAG.
// This visitor traverses an ArithmeticDAG and computes the maximum
// multiplicative depth required, which determines the noise budget and
// parameters needed for FHE schemes.
//
// Multiplicative depth is the maximum number of sequential multiplications
// in the longest path from inputs to outputs. Addition and rotation do not
// increase depth.
class MultiplicativeDepthVisitor {
 public:
  using NodeTy = ArithmeticDagNode<SymbolicValue>;

  MultiplicativeDepthVisitor() {}

  // Main entry point - computes multiplicative depth of the DAG
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
  // Cache to store computed depth for each node (memoization)
  std::unordered_map<const NodeTy*, int64_t> nodeDepth_;

  // Track visited nodes for cycle detection
  std::unordered_set<const NodeTy*> visitedNodes_;

  const NodeTy* currentNode_ = nullptr;

  // Internal recursive traversal with caching
  int64_t processInternal(const std::shared_ptr<NodeTy>& node);
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_MULTIPLICATIVEDEPTHVISITOR_H_
