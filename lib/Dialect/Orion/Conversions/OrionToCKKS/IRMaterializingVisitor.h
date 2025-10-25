#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
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

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level);

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

  Value maybeRescale(ImplicitLocOpBuilder builder, Value value,
                     lwe::LWECiphertextType resultType, bool rescale) {
    Value result = value;
    if (rescale) {
      FailureOr<lwe::LWECiphertextType> ctTypeResult =
          applyModReduce(resultType);
      if (failed(ctTypeResult)) {
        emitError(result.getLoc())
            << "Cannot rescale ciphertext type: " << resultType;
        return Value();
      }
      auto ctType = ctTypeResult.value();
      result = ckks::RescaleOp::create(builder, ctType, result,
                                       ctType.getCiphertextSpace().getRing());
    }
    return result;
  }

  template <typename T, typename CtCtOp, typename CtPtOp>
  Value binop(const T& node, bool rescale = false) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);
    return TypeSwitch<Type, Value>(lhs.getType())
        .template Case<lwe::LWECiphertextType>([&](auto ty) {
          if (isa<RankedTensorType>(rhs.getType())) {
            lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
                builder.getContext(), ty.getApplicationData(),
                ty.getPlaintextSpace());
            auto encodedRhs = lwe::RLWEEncodeOp::create(
                builder, ptTy, rhs, ty.getPlaintextSpace().getEncoding(),
                ty.getPlaintextSpace().getRing());
            return maybeRescale(
                builder, CtPtOp::create(builder, lhs, encodedRhs).getResult(),
                ty, rescale);
          }

          if (!isa<lwe::LWECiphertextType>(rhs.getType())) {
            emitError(lhs.getLoc()) << "Unsupported types for binary operation "
                                       "(lhs is ciphertext): \n\nlhs="
                                    << lhs << "\n\nrhs=" << rhs << "\n";
            return Value();
          }
          return maybeRescale(builder,
                              CtCtOp::create(builder, lhs, rhs).getResult(), ty,
                              rescale);
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
          return maybeRescale(
              builder, CtPtOp::create(builder, encodedLhs, rhs).getResult(),
              ctTy, rescale);
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
