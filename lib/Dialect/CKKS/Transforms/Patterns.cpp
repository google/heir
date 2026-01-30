#include "lib/Dialect/CKKS/Transforms/Patterns.h"

#include <cstdint>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

LogicalResult DecomposeRelinearizePattern::matchAndRewrite(
    RelinearizeOp op, PatternRewriter& rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  if (!op.getKeySwitchingKey()) {
    return rewriter.notifyMatchFailure(op, "no key switching key provided");
  }
  lwe::LWECiphertextType ctType =
      dyn_cast<lwe::LWECiphertextType>(op.getInput().getType());
  if (ctType.getCiphertextSpace().getSize() != 3) {
    return rewriter.notifyMatchFailure(
        op, "ciphertext must have exactly three components");
  }

  Value input0 = ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(0));
  Value input1 = ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(1));
  Value input2 = ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(2));

  polynomial::RingAttr ringAttr = ctType.getCiphertextSpace().getRing();
  lwe::CiphertextSpaceAttr ctAttr = lwe::CiphertextSpaceAttr::get(
      op.getContext(), ringAttr,
      ctType.getCiphertextSpace().getEncryptionType(), 2);
  lwe::LWECiphertextType outCTType = lwe::LWECiphertextType::get(
      op.getContext(), ctType.getApplicationData(), ctType.getPlaintextSpace(),
      ctAttr, ctType.getKey(), ctType.getModulusChain());
  KeySwitchInnerOp ksPoly =
      KeySwitchInnerOp::create(b, input2, op.getKeySwitchingKey());
  Value ksCT = FromCoeffsOp::create(
      b, outCTType, {ksPoly.getConstTerm(), ksPoly.getLinearTerm()});
  Value linearCT = FromCoeffsOp::create(b, outCTType, {input0, input1});
  Value sum = AddOp::create(b, ksCT, linearCT);
  rewriter.replaceOp(op, sum);
  return success();
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
