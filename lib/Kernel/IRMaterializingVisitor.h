#ifndef LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
#define LIB_KERNEL_IRMATERIALIZINGVISITOR_H_

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

// Walks the arithmetic DAG and generates MLIR for it.
class IRMaterializingVisitor : public CachingVisitor<SSAValue, Value> {
 public:
  using CachingVisitor<SSAValue, Value>::operator();

  IRMaterializingVisitor(ImplicitLocOpBuilder& builder, Type evaluatedType)
      : CachingVisitor<SSAValue, Value>(),
        builder(builder),
        evaluatedType(evaluatedType) {}

  template <typename T, typename FloatOp, typename IntOp>
  Value binop(const T& node) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);
    return TypeSwitch<Type, Value>(getElementTypeOrSelf(evaluatedType))
        .template Case<FloatType>(
            [&](auto ty) { return FloatOp::create(builder, lhs, rhs); })
        .template Case<IntegerType>(
            [&](auto ty) { return IntOp::create(builder, lhs, rhs); })
        .Default([&](Type) {
          llvm_unreachable("Unsupported type for binary operation");
          return Value();
        });
  }

  Value operator()(const ConstantNode& node) override;
  Value operator()(const LeafNode<SSAValue>& node) override;
  Value operator()(const AddNode<SSAValue>& node) override;
  Value operator()(const SubtractNode<SSAValue>& node) override;
  Value operator()(const MultiplyNode<SSAValue>& node) override;
  Value operator()(const LeftRotateNode<SSAValue>& node) override;
  Value operator()(const ExtractNode<SSAValue>& node) override;

 private:
  ImplicitLocOpBuilder& builder;
  Type evaluatedType;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_IRMATERIALIZINGVISITOR_H_
