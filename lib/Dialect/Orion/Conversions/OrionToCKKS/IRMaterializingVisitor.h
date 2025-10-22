#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_

#include "lib/Dialect/LWE/IR/LWETypes.h"
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
namespace orion {

// Walks the arithmetic DAG and generates MLIR for it.
class IRMaterializingVisitor
    : public kernel::CachingVisitor<kernel::SSAValue, Value> {
 public:
  using CachingVisitor<kernel::SSAValue, Value>::operator();

  IRMaterializingVisitor(ImplicitLocOpBuilder& builder)
      : CachingVisitor<kernel::SSAValue, Value>(), builder(builder) {}

  template <typename T, typename CtCtOp, typename CtPtOp>
  Value binop(const T& node) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);
    return TypeSwitch<Type, Value>(getElementTypeOrSelf(lhs.getType()))
        .template Case<lwe::LWECiphertextType>([&](auto ty) {
          if (isa<lwe::LWEPlaintextType>(rhs.getType())) {
            return CtPtOp::create(builder, lhs, rhs).getResult();
          }

          assert(isa<lwe::LWECiphertextType>(rhs.getType()));
          return CtCtOp::create(builder, lhs, rhs).getResult();
        })
        .template Case<lwe::LWEPlaintextType>([&](auto ty) {
          // There are no plaintext-plaintext ops
          assert(isa<lwe::LWECiphertextType>(rhs.getType()));
          return CtPtOp::create(builder, lhs, rhs).getResult();
        })
        .Default([&](Type) {
          emitError(lhs.getLoc())
              << "Unsupported types for binary operation: \n\nlhs="
              << lhs.getType() << "\n\nrhs=" << lhs.getType() << "\n";
          return Value();
        });
  }

  Value operator()(const kernel::ConstantScalarNode& node) override;
  Value operator()(const kernel::ConstantTensorNode& node) override;
  Value operator()(const kernel::LeafNode<kernel::SSAValue>& node) override;
  Value operator()(const kernel::AddNode<kernel::SSAValue>& node) override;
  Value operator()(const kernel::SubtractNode<kernel::SSAValue>& node) override;
  Value operator()(const kernel::MultiplyNode<kernel::SSAValue>& node) override;
  Value operator()(
      const kernel::LeftRotateNode<kernel::SSAValue>& node) override;
  Value operator()(const kernel::ExtractNode<kernel::SSAValue>& node) override;

 private:
  ImplicitLocOpBuilder& builder;
};

}  // namespace orion
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
