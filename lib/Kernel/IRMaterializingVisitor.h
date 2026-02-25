#ifndef LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
#define LIB_KERNEL_IRMATERIALIZINGVISITOR_H_

#include <functional>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
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
class IRMaterializingVisitor
    : public CachingVisitor<SSAValue, std::vector<Value>> {
 public:
  using CachingVisitor<SSAValue, std::vector<Value>>::operator();

  IRMaterializingVisitor(
      ImplicitLocOpBuilder& builder, Type evaluatedType,
      const std::function<void(Operation*)>& createdOpCallback =
          [](Operation* op) {})
      : CachingVisitor<SSAValue, std::vector<Value>>(),
        builder(builder),
        evaluatedType(evaluatedType),
        createdOpCallback(createdOpCallback) {}

  template <typename T, typename FloatOp, typename IntOp>
  std::vector<Value> binop(const T& node) {
    Value lhs = this->process(node.left)[0];
    Value rhs = this->process(node.right)[0];
    auto op = TypeSwitch<Type, Operation*>(getElementTypeOrSelf(evaluatedType))
                  .template Case<mlir::FloatType>([&](auto ty) {
                    return FloatOp::create(builder, lhs, rhs);
                  })
                  .template Case<mlir::IntegerType>(
                      [&](auto ty) { return IntOp::create(builder, lhs, rhs); })
                  .Default([&](Type) {
                    llvm_unreachable("Unsupported type for binary operation");
                    return nullptr;
                  });
    createdOpCallback(op);
    return {op->getResult(0)};
  }

  std::vector<Value> operator()(const ConstantScalarNode& node) override;
  std::vector<Value> operator()(const ConstantTensorNode& node) override;
  std::vector<Value> operator()(const LeafNode<SSAValue>& node) override;
  std::vector<Value> operator()(const AddNode<SSAValue>& node) override;
  std::vector<Value> operator()(const SubtractNode<SSAValue>& node) override;
  std::vector<Value> operator()(const MultiplyNode<SSAValue>& node) override;
  std::vector<Value> operator()(const LeftRotateNode<SSAValue>& node) override;
  std::vector<Value> operator()(const ExtractNode<SSAValue>& node) override;

 private:
  ImplicitLocOpBuilder& builder;
  Type evaluatedType;
  const std::function<void(Operation*)> createdOpCallback;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
