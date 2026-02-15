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
      ImplicitLocOpBuilder& builder,
      const std::function<void(Operation*)>& createdOpCallback =
          [](Operation* op) {})
      : builder(builder),
        createdOpCallback(createdOpCallback) {}

  // Main entry point: process a single DAG node.
  // The output of the dag is enforced to be a single value, though
  // internal nodes may return multiple values.
  Value process(const std::shared_ptr<ArithmeticDagNode<SSAValue>>& node) {
    std::vector<Value> results = processInternal(node);
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
      for (Value v : processInternal(node)) {
        results.push_back(v);
      }
    }
    return results;
  }

  // Visitor operator() overloads for each node type
  std::vector<Value> operator()(const ConstantScalarNode& node);
  std::vector<Value> operator()(const SplatNode& node);
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
  // Internal process method with caching
  std::vector<Value> processInternal(
      const std::shared_ptr<ArithmeticDagNode<SSAValue>>& node);

  // Helper to ensure a value has index type, inserting a cast if needed
  Value ensureIndexType(Value val);

  // Helper to normalize shapes for binary operations
  // If one operand is [1, N] and the other is [N], expand [N] to [1, N]
  std::pair<Value, Value> normalizeShapes(Value lhs, Value rhs);

  // Helper for binary operations
  template <typename T, typename FloatOp, typename IntOp>
  std::vector<Value> binop(const T& node) {
    // Process operands
    std::vector<Value> lhsVals = processInternal(node.left);
    std::vector<Value> rhsVals = processInternal(node.right);
    assert(lhsVals.size() == 1 && rhsVals.size() == 1 &&
           "Binary operation operands must materialize to single values");
    Value lhs = lhsVals[0];
    Value rhs = rhsVals[0];

    // Normalize shapes to handle [1, N] vs [N] mismatches
    auto [normalizedLhs, normalizedRhs] = normalizeShapes(lhs, rhs);

    // Infer operation type from the operands
    auto op = TypeSwitch<Type, Operation*>(getElementTypeOrSelf(normalizedLhs.getType()))
                  .template Case<mlir::FloatType>([&](auto ty) {
                    return FloatOp::create(builder, normalizedLhs, normalizedRhs);
                  })
                  .template Case<mlir::IntegerType, mlir::IndexType>(
                      [&](auto ty) { return IntOp::create(builder, normalizedLhs, normalizedRhs); })
                  .Default([&](Type) {
                    llvm_unreachable("Unsupported type for binary operation");
                    return nullptr;
                  });
    createdOpCallback(op);
    return {op->getResult(0)};
  }

  ImplicitLocOpBuilder& builder;
  const std::function<void(Operation*)> createdOpCallback;

  // Simple cache: keys are node pointers
  std::unordered_map<const ArithmeticDagNode<SSAValue>*, std::vector<Value>>
      cache;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
