#ifndef LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
#define LIB_KERNEL_IRMATERIALIZINGVISITOR_H_

#include <functional>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// Walks the arithmetic DAG and generates MLIR for it. Optionally applies
// createdOpCallback to each created operation.
class IRMaterializingVisitor
    : public CachingVisitor<SSAValue, std::vector<Value>,
                            ImplicitLocOpBuilder&> {
 public:
  using CachingVisitor<SSAValue, std::vector<Value>,
                       ImplicitLocOpBuilder&>::operator();

  IRMaterializingVisitor(
      Type evaluatedType,
      const std::function<void(Operation*)>& createdOpCallback =
          [](Operation* op) {})
      : CachingVisitor<SSAValue, std::vector<Value>, ImplicitLocOpBuilder&>(),
        evaluatedType(evaluatedType),
        createdOpCallback(createdOpCallback) {}

  template <typename T, typename FloatOp, typename IntOp>
  std::vector<Value> binop(const T& node, ImplicitLocOpBuilder& builder) {
    Value lhs = this->process(node.left, builder)[0];
    Value rhs = this->process(node.right, builder)[0];
    auto op = TypeSwitch<Type, Operation*>(getElementTypeOrSelf(lhs.getType()))
                  .template Case<mlir::FloatType>([&](auto ty) {
                    return FloatOp::create(builder, lhs, rhs);
                  })
                  .template Case<mlir::IntegerType, mlir::IndexType>(
                      [&](auto ty) { return IntOp::create(builder, lhs, rhs); })
                  .Default([&](Type) {
                    llvm_unreachable("Unsupported type for binary operation");
                    return nullptr;
                  });
    createdOpCallback(op);
    return {op->getResult(0)};
  }

  std::vector<Value> operator()(const ConstantScalarNode& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const SplatNode& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const ConstantTensorNode& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const LeafNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const AddNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const SubtractNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const MultiplyNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const FloorDivNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const LeftRotateNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const ExtractNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const InsertNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const ComparisonNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const IfElseNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const VariableNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const ForLoopNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const YieldNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;
  std::vector<Value> operator()(const ResultAtNode<SSAValue>& node,
                                ImplicitLocOpBuilder& builder) override;

 private:
  Type evaluatedType;
  const std::function<void(Operation*)> createdOpCallback;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
