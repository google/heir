#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

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

  IRMaterializingVisitor(ImplicitLocOpBuilder& builder,
                         lwe::LWEPlaintextType ptTy)
      : CachingVisitor<kernel::SSAValue, Value>(),
        builder(builder),
        plaintextType(ptTy) {}

  Value maybeRescale(Value value, lwe::LWECiphertextType resultType,
                     bool rescale) {
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

  Value encodeCleartextOperand(lwe::LWECiphertextType ctTy, Value cleartext) {
    lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
        builder.getContext(), ctTy.getApplicationData(),
        ctTy.getPlaintextSpace());
    return lwe::RLWEEncodeOp::create(builder, ptTy, cleartext,
                                     ctTy.getPlaintextSpace().getEncoding(),
                                     ctTy.getPlaintextSpace().getRing())
        .getResult();
  }

  template <typename T, typename CtCtOp, typename CtPtOp, typename CleartextOp>
  Value binop(const T& node, bool rescale = false) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);

    return TypeSwitch<Type, Value>(lhs.getType())
        .template Case<lwe::LWECiphertextType>([&](auto ty) {
          if (isa<RankedTensorType>(rhs.getType())) {
            return maybeRescale(
                CtPtOp::create(builder, lhs, encodeCleartextOperand(ty, rhs))
                    .getResult(),
                ty, rescale);
          }

          if (isa<lwe::LWEPlaintextType>(rhs.getType())) {
            return maybeRescale(CtPtOp::create(builder, lhs, rhs).getResult(),
                                ty, rescale);
          }

          return maybeRescale(CtCtOp::create(builder, lhs, rhs).getResult(), ty,
                              rescale);
        })
        .template Case<lwe::LWEPlaintextType>([&](auto ty) {
          if (isa<RankedTensorType>(rhs.getType())) {
            auto encodeOp = dyn_cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
            return CleartextOp::create(builder, encodeOp.getInput(), rhs)
                .getResult();
          }

          if (isa<lwe::LWEPlaintextType>(rhs.getType())) {
            auto encodedLhs = dyn_cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
            auto encodedRhs = dyn_cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
            return CleartextOp::create(builder, encodedLhs.getInput(),
                                       encodedRhs.getInput())
                .getResult();
          }

          auto ctTy = dyn_cast<lwe::LWECiphertextType>(rhs.getType());
          if (!ctTy) {
            emitError(lhs.getLoc())
                << "Unsupported types for binary operation: \n\nlhs=" << lhs
                << "\n\nrhs=" << rhs << "\n";
            return Value();
          }
          return maybeRescale(CtPtOp::create(builder, lhs, rhs).getResult(),
                              ctTy, rescale);
        })
        .template Case<RankedTensorType>([&](auto ty) {
          auto ctTy = cast<lwe::LWECiphertextType>(rhs.getType());
          return maybeRescale(
              CtPtOp::create(builder, encodeCleartextOperand(ctTy, lhs), rhs)
                  .getResult(),
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
  lwe::LWEPlaintextType plaintextType;
};

}  // namespace orion
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
