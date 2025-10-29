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
// meant to handle quirks of the Orion import process. For example it
// handles scale encoding of cleartexts, and special cases for tensors of
// ciphertexts vs ciphertext-semantic tensors of cleartexts. The former
// requires simple ops like tensor.extract and ckks.rotate, while the latter
// requires tensor.extract_slice and tensor_ext.rotate.
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

  // Relinearize and rescale the ciphertext if relinAndRescale is true.
  Value relinAndRescale(Value value, lwe::LWECiphertextType resultType,
                        bool relinearize, bool rescale) {
    Value result = value;
    if (relinearize) {
      auto inputDimension = cast<lwe::LWECiphertextType>(value.getType())
                                .getCiphertextSpace()
                                .getSize();
      SmallVector<int32_t> fromBasis;
      for (int i = 0; i < inputDimension; ++i) {
        fromBasis.push_back(i);
      }
      SmallVector<int32_t> toBasis = {0, 1};
      auto relinOp = ckks::RelinearizeOp::create(
          builder, result, builder.getDenseI32ArrayAttr(fromBasis),
          builder.getDenseI32ArrayAttr(toBasis));
      result = relinOp.getResult();
    }
    if (rescale) {
      FailureOr<lwe::LWECiphertextType> ctTypeResult =
          applyModReduce(resultType);
      if (failed(ctTypeResult)) {
        emitError(result.getLoc())
            << "Cannot rescale ciphertext type, inserting extra bootstrap op";
        // sub 1 because the max level is the last index in the chain.
        int64_t maxLevel =
            resultType.getModulusChain().getElements().size() - 1;

        // Now we cheat a little bit: normally bootstrap itself would consume
        // some levels, which depends on the chosen backend. In our case, we're
        // lowering to library backends that handle this opaquely.
        //
        // TODO(#1207): fix if this pass still matters when lowering to
        // polynomial.
        FailureOr<lwe::LWECiphertextType> outputTypeResult =
            cloneAtLevel(resultType, maxLevel);
        if (failed(outputTypeResult)) {
          emitError(result.getLoc()) << "Failed to insert bootstrap";
          return Value();
        }
        result = ckks::BootstrapOp::create(builder, outputTypeResult.value(),
                                           result);
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
  Value binop(const T& node) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);

    bool relinearize =
        static_cast<bool>(std::is_same<CtCtOp, ckks::MulOp>::value);
    bool rescale =
        static_cast<bool>(std::is_same<CtCtOp, ckks::MulOp>::value) ||
        static_cast<bool>(std::is_same<CtPtOp, ckks::MulPlainOp>::value);

    return TypeSwitch<Type, Value>(lhs.getType())
        .template Case<lwe::LWECiphertextType>([&](auto ty) {
          if (isa<RankedTensorType>(rhs.getType())) {
            return relinAndRescale(
                CtPtOp::create(builder, lhs, encodeCleartextOperand(ty, rhs))
                    .getResult(),
                ty, relinearize, rescale);
          }

          if (isa<lwe::LWEPlaintextType>(rhs.getType())) {
            return relinAndRescale(
                CtPtOp::create(builder, lhs, rhs).getResult(), ty, relinearize,
                rescale);
          }

          return relinAndRescale(CtCtOp::create(builder, lhs, rhs).getResult(),
                                 ty, relinearize, rescale);
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
          return relinAndRescale(CtPtOp::create(builder, lhs, rhs).getResult(),
                                 ctTy, relinearize, rescale);
        })
        .template Case<RankedTensorType>([&](auto ty) {
          auto ctTy = cast<lwe::LWECiphertextType>(rhs.getType());
          return relinAndRescale(
              CtPtOp::create(builder, encodeCleartextOperand(ctTy, lhs), rhs)
                  .getResult(),
              ctTy, relinearize, rescale);
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
