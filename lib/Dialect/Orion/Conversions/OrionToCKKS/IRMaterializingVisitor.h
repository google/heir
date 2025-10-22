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
// meant to handle quirks of the Orion import process. For example it
// handles scale encoding of cleartexts, and special cases for tensors of
// ciphertexts vs ciphertext-semantic tensors of cleartexts. The former
// requires simple ops like tensor.extract and ckks.rotate, while the latter
// requires tensor.extract_slice and tensor_ext.rotate.
class IRMaterializingVisitor
    : public kernel::CachingVisitor<kernel::SSAValue, Value> {
 public:
  using CachingVisitor<kernel::SSAValue, Value>::operator();

  IRMaterializingVisitor(ImplicitLocOpBuilder& builder,
                         lwe::LWEPlaintextType ptTy, bool rescaleAfterCtPtMul,
                         int64_t logDefaultScale)
      : CachingVisitor<kernel::SSAValue, Value>(),
        builder(builder),
        plaintextType(ptTy),
        rescaleAfterCtPtMul(rescaleAfterCtPtMul),
        logDefaultScale(logDefaultScale) {}

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

  // Encode a cleartext so that its scaling factor matches the ctTy encoding's
  // scaling factor.
  Value encodeCleartextOperandToMatchCt(lwe::LWECiphertextType ctTy,
                                        Value cleartext) {
    MLIRContext* ctx = builder.getContext();
    int64_t logScale = lwe::getScalingFactorFromEncodingAttr(
        ctTy.getPlaintextSpace().getEncoding());
    auto encoding = lwe::InverseCanonicalEncodingAttr::get(ctx, logScale);
    auto ring = ctTy.getPlaintextSpace().getRing();

    lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
        ctx, ctTy.getApplicationData(),
        lwe::PlaintextSpaceAttr::get(ctx, ring, encoding));
    return lwe::RLWEEncodeOp::create(builder, ptTy, cleartext, encoding, ring)
        .getResult();
  }

  // Encode a cleartext so that its scaling factor matches the expected scaling
  // factor if the ctTy is multiplied by a plaintext with the given scaling
  // factor. This depends on how the target backend handles rescaling.
  Value encodeCleartextOperandToMatchCtMulOutput(lwe::LWECiphertextType ctTy,
                                                 Value cleartext,
                                                 int64_t logScaleOperand) {
    MLIRContext* ctx = builder.getContext();
    // TODO(#2364) without high-precision scale management, we can't
    // use the exact modulus from the modulus chain.
    int64_t currentLogModulus = static_cast<int64_t>(std::ceil(
        std::log2(ctTy.getModulusChain()
                      .getElements()[ctTy.getModulusChain().getCurrent()]
                      .getInt())));
    int64_t currentLogScale = lwe::getScalingFactorFromEncodingAttr(
        ctTy.getPlaintextSpace().getEncoding());
    int64_t logScale = rescaleAfterCtPtMul
                           ? logDefaultScale
                           : currentLogModulus + currentLogScale;
    auto encoding = lwe::InverseCanonicalEncodingAttr::get(ctx, logScale);
    auto ring = ctTy.getPlaintextSpace().getRing();

    lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
        ctx, ctTy.getApplicationData(),
        lwe::PlaintextSpaceAttr::get(ctx, ring, encoding));
    return lwe::RLWEEncodeOp::create(builder, ptTy, cleartext, encoding, ring)
        .getResult();
  }

  template <typename T, typename CtCtOp, typename CtPtOp, typename CleartextOp>
  Value binop(const T& node) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);
    bool isMul = static_cast<bool>(std::is_same<CtCtOp, ckks::MulOp>::value);

    return TypeSwitch<Type, Value>(lhs.getType())
        .template Case<lwe::LWECiphertextType>([&](auto ty) {
          if (isa<RankedTensorType>(rhs.getType())) {
            // Ciphertext-plaintext
            bool rescale = isMul && rescaleAfterCtPtMul;
            return relinAndRescale(
                CtPtOp::create(builder, lhs,
                               encodeCleartextOperandToMatchCt(ty, rhs))
                    .getResult(),
                ty, /*relinearize=*/false, rescale);
          }

          if (isa<lwe::LWEPlaintextType>(rhs.getType())) {
            auto rhsEncodeOp = cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
            bool rescale = isMul && rescaleAfterCtPtMul;
            auto newRhs =
                encodeCleartextOperandToMatchCt(ty, rhsEncodeOp.getInput());
            return relinAndRescale(
                CtPtOp::create(builder, lhs, newRhs).getResult(), ty,
                /*relinearize=*/false, /*rescale=*/rescale);
          }

          // Ciphertext-ciphertext
          return relinAndRescale(CtCtOp::create(builder, lhs, rhs).getResult(),
                                 ty, /*relinearize=*/isMul, /*rescale=*/isMul);
        })

        .template Case<lwe::LWEPlaintextType>([&](auto ty) {
          auto encodedLhs = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
          if (isa<RankedTensorType>(rhs.getType())) {
            return CleartextOp::create(builder, encodedLhs.getInput(), rhs)
                .getResult();
          }

          if (isa<lwe::LWEPlaintextType>(rhs.getType())) {
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
          bool rescale = isMul && rescaleAfterCtPtMul;
          auto lhsEncodeOp = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
          auto newLhs =
              encodeCleartextOperandToMatchCt(ctTy, lhsEncodeOp.getInput());
          return relinAndRescale(
              CtPtOp::create(builder, newLhs, rhs).getResult(), ctTy,
              /*relinearize=*/false, rescale);
        })
        .template Case<RankedTensorType>([&](auto ty) {
          auto ctTy = cast<lwe::LWECiphertextType>(rhs.getType());
          bool rescale = isMul && rescaleAfterCtPtMul;
          return relinAndRescale(
              CtPtOp::create(builder,
                             encodeCleartextOperandToMatchCt(ctTy, lhs), rhs)
                  .getResult(),
              ctTy, /*relinearize=*/false, rescale);
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
  bool rescaleAfterCtPtMul;
  int64_t logDefaultScale;
};

}  // namespace orion
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
