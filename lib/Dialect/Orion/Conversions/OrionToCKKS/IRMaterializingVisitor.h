#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace orion {

// Walks the arithmetic DAG and generates MLIR for it. This materializer is
// weird because some SSA values are tensors of ciphertexts while others are
// ciphertext-semantic tensors of cleartexts. The former requires simple ops
// like tensor.extract and ckks.rotate, while the latter requires
// tensor.extract_slice and tensor_ext.rotate.
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
    return TypeSwitch<Type, Value>(lhs.getType())
        .template Case<lwe::LWECiphertextType>([&](auto ty) {
          if (isa<RankedTensorType>(rhs.getType())) {
            // FIXME: (here and below) set the right scaling factor for the
            // result plaintextType. Needs to get passed through from the input
            // op!
            lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
                builder.getContext(), ty.getApplicationData(),
                ty.getPlaintextSpace());
            auto encodedRhs = lwe::RLWEEncodeOp::create(
                builder, ptTy, rhs, ty.getPlaintextSpace().getEncoding(),
                ty.getPlaintextSpace().getRing());
            return CtPtOp::create(builder, lhs, encodedRhs).getResult();
          }

          if (!isa<lwe::LWECiphertextType>(rhs.getType())) {
            emitError(lhs.getLoc()) << "Unsupported types for binary operation "
                                       "(lhs is ciphertext): \n\nlhs="
                                    << lhs << "\n\nrhs=" << rhs << "\n";
            return Value();
          }
          return CtCtOp::create(builder, lhs, rhs).getResult();
        })
        .template Case<RankedTensorType>([&](auto ty) {
          auto ctTy = dyn_cast<lwe::LWECiphertextType>(rhs.getType());
          if (!ctTy) {
            emitError(lhs.getLoc()) << "Unsupported types for binary operation "
                                       "(lhs is cleartext): \n\nlhs="
                                    << lhs << "\n\nrhs=" << rhs << "\n";
            return Value();
          }
          lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
              builder.getContext(), ctTy.getApplicationData(),
              ctTy.getPlaintextSpace());
          auto encodedLhs = lwe::RLWEEncodeOp::create(
              builder, ptTy, lhs, ctTy.getPlaintextSpace().getEncoding(),
              ctTy.getPlaintextSpace().getRing());
          return CtPtOp::create(builder, encodedLhs, rhs).getResult();
        })
        .Default([&](Type) {
          emitError(lhs.getLoc())
              << "Unsupported types for binary operation: \n\nlhs=" << lhs
              << "\n\nrhs=" << rhs << "\n";
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
