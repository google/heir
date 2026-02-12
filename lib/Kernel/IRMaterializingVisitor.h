#ifndef LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
#define LIB_KERNEL_IRMATERIALIZINGVISITOR_H_

#include <cassert>
#include <functional>
#include <unordered_map>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// Walks the arithmetic DAG and generates MLIR for it. Optionally applies
// createdOpCallback to each created operation.
//
// This visitor uses type-aware caching, allowing the same DAG node to be
// materialized to different MLIR types based on context (e.g., an index
// constant can be materialized as either index type or tensor type).
class IRMaterializingVisitor {
 public:
  IRMaterializingVisitor(
      ImplicitLocOpBuilder& builder, Type evaluatedType,
      const std::function<void(Operation*)>& createdOpCallback =
          [](Operation* op) {})
      : builder(builder),
        evaluatedType(evaluatedType),
        currentType(evaluatedType),
        createdOpCallback(createdOpCallback) {}

  // Main entry point: process a single DAG node with default type.
  // The output of the dag is enforced to be a single value, though
  // internal nodes may return multiple values.
  Value process(
      const std::shared_ptr<ArithmeticDagNode<SSAValue>>& node) {
    std::vector<Value> results = process(node, nullptr);
    assert(results.size() == 1 &&
           "Expected DAG to materialize to a single value, but got multiple");
    return results[0];
  }

  // Process multiple DAG nodes
  std::vector<Value> process(
      llvm::ArrayRef<std::shared_ptr<ArithmeticDagNode<SSAValue>>> nodes) {
    std::vector<Value> results;
    results.reserve(nodes.size());
    for (const auto& node : nodes) {
      for (Value v : process(node, nullptr)) {
        results.push_back(v);
      }
    }
    return results;
  }

  // Type-aware process method that allows specifying the expected type for
  // materialization. If type is null, uses evaluatedType.
  std::vector<Value> process(
      const std::shared_ptr<ArithmeticDagNode<SSAValue>>& node, Type type);

  // Visitor operator() overloads for each node type
  std::vector<Value> operator()(const ConstantScalarNode& node);
  std::vector<Value> operator()(const ConstantTensorNode& node);
  std::vector<Value> operator()(const LeafNode<SSAValue>& node);
  std::vector<Value> operator()(const AddNode<SSAValue>& node);
  std::vector<Value> operator()(const SubtractNode<SSAValue>& node);
  std::vector<Value> operator()(const MultiplyNode<SSAValue>& node);
  std::vector<Value> operator()(const DivideNode<SSAValue>& node);
  std::vector<Value> operator()(const PowerNode<SSAValue>& node);
  std::vector<Value> operator()(const LeftRotateNode<SSAValue>& node);
  std::vector<Value> operator()(const ExtractNode<SSAValue>& node);
  std::vector<Value> operator()(const VariableNode<SSAValue>& node);
  std::vector<Value> operator()(const ForLoopNode<SSAValue>& node);
  std::vector<Value> operator()(const YieldNode<SSAValue>& node);
  std::vector<Value> operator()(const ResultAtNode<SSAValue>& node);

 private:
  // Helper for binary operations
  template <typename T, typename FloatOp, typename IntOp>
  std::vector<Value> binop(const T& node) {
    Value lhs = process(node.left, currentType);
    Value rhs = process(node.right, currentType);
    auto op = TypeSwitch<Type, Operation*>(getElementTypeOrSelf(currentType))
                  .template Case<FloatType>([&](auto ty) {
                    return FloatOp::create(builder, lhs, rhs);
                  })
                  .template Case<IntegerType, IndexType>(
                      [&](auto ty) { return IntOp::create(builder, lhs, rhs); })
                  .Default([&](Type) {
                    llvm_unreachable("Unsupported type for binary operation");
                    return nullptr;
                  });
    createdOpCallback(op);
    return {op->getResult(0)};
  }

  // Hash function for (node pointer, type) pairs used as cache keys
  struct NodeTypePairHash {
    std::size_t operator()(
        const std::pair<const ArithmeticDagNode<SSAValue>*, Type>& p) const {
      return std::hash<const void*>()(p.first) ^
             (std::hash<const void*>()(p.second.getAsOpaquePointer()) << 1);
    }
  };

  ImplicitLocOpBuilder& builder;
  Type evaluatedType;
  Type currentType;  // The type to use for the current node being visited
  const std::function<void(Operation*)> createdOpCallback;

  // Type-aware cache: keys are (node pointer, type) pairs
  std::unordered_map<std::pair<const ArithmeticDagNode<SSAValue>*, Type>,
                     std::vector<Value>, NodeTypePairHash>
      typeAwareCache;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
