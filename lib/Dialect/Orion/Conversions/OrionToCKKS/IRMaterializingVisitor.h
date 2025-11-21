#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_IRMATERIALIZINGVISITOR_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/Support/Debug.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

#define DEBUG_TYPE "orion-to-ckks"

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

  // Relinearize and rescale the ciphertext based on the given options. This
  // always drops the modulus chain by one level.
  Value relinAndRescale(Value value, bool relinearize, bool rescale);

  // Encode a cleartext so that its scaling factor matches the ctTy encoding's
  // scaling factor.
  Value encodeCleartextOperand(lwe::LWECiphertextType ctTy, Value cleartext,
                               bool useDefaultScale = false);

  template <typename T, typename CtCtOp, typename CtPtOp, typename CleartextOp>
  Value nonMulBinop(const T& node) {
    Value lhs = this->process(node.left);
    Value rhs = this->process(node.right);

    // Just dyn_cast to all possibilities, and keep the nesting structure flat
    // to avoid awkward contortions of doing type switches on both lhs and rhs.
    lwe::LWECiphertextType lhsCiphertextType =
        dyn_cast<lwe::LWECiphertextType>(lhs.getType());
    lwe::LWECiphertextType rhsCiphertextType =
        dyn_cast<lwe::LWECiphertextType>(rhs.getType());
    lwe::LWEPlaintextType lhsPlaintextType =
        dyn_cast<lwe::LWEPlaintextType>(lhs.getType());
    lwe::LWEPlaintextType rhsPlaintextType =
        dyn_cast<lwe::LWEPlaintextType>(rhs.getType());
    RankedTensorType lhsTensorType = dyn_cast<RankedTensorType>(lhs.getType());
    RankedTensorType rhsTensorType = dyn_cast<RankedTensorType>(rhs.getType());

    // Plaintext-Cleartext case
    if (lhsPlaintextType && rhsTensorType) {
      auto encodedLhs = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
      return CleartextOp::create(builder, encodedLhs.getInput(), rhs)
          .getResult();
    }
    if (lhsTensorType && rhsPlaintextType) {
      auto encodedRhs = cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
      return CleartextOp::create(builder, lhs, encodedRhs.getInput())
          .getResult();
    }

    // Plaintext-plaintext case
    if (lhsPlaintextType && rhsPlaintextType) {
      auto encodedLhs = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
      auto encodedRhs = cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
      auto cleartextOp = CleartextOp::create(builder, encodedLhs.getInput(),
                                             encodedRhs.getInput());
      auto encodeOp = lwe::RLWEEncodeOp::create(
          builder, lhsPlaintextType, cleartextOp.getResult(),
          lhsPlaintextType.getPlaintextSpace().getEncoding(),
          lhsPlaintextType.getPlaintextSpace().getRing());
      return encodeOp;
    }

    // Ciphertext-Plaintext case
    if (lhsCiphertextType && rhsPlaintextType) {
      auto rhsEncodeOp = cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
      auto newRhs =
          encodeCleartextOperand(lhsCiphertextType, rhsEncodeOp.getInput());
      auto ctPtOp = CtPtOp::create(builder, lhs, newRhs);
      return ctPtOp.getResult();
    }
    if (lhsPlaintextType && rhsCiphertextType) {
      auto lhsEncodeOp = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
      auto newLhs =
          encodeCleartextOperand(rhsCiphertextType, lhsEncodeOp.getInput());
      auto ctPtOp = CtPtOp::create(builder, newLhs, rhs);
      return ctPtOp.getResult();
    }

    // Ciphertext-ciphertext case
    if (lhsCiphertextType && rhsCiphertextType) {
      auto ctCtOp = CtCtOp::create(builder, lhs, rhs).getResult();
      return ctCtOp;
    }

    emitError(lhs.getLoc())
        << "Unsupported types for non-mul binary operation: \n\nlhs=" << lhs
        << "\n\nrhs=" << rhs << "\n";
    return Value();
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
