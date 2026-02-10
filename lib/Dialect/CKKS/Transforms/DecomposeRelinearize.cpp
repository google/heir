#include "lib/Dialect/CKKS/Transforms/DecomposeRelinearize.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DEF_DECOMPOSERELINEARIZE
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

LogicalResult DecomposeRelinearizePattern::matchAndRewrite(
    RelinearizeOp op, PatternRewriter& rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  if (!op.getKeySwitchingKey()) {
    return rewriter.notifyMatchFailure(op, "no key switching key provided");
  }
  auto ctType = dyn_cast<lwe::LWECiphertextType>(op.getInput().getType());
  if (ctType.getCiphertextSpace().getSize() != 3) {
    return rewriter.notifyMatchFailure(
        op, "ciphertext must have exactly three components");
  }

  Value input0 =
      lwe::ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(0));
  Value input1 =
      lwe::ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(1));
  Value input2 =
      lwe::ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(2));

  polynomial::RingAttr ringAttr = ctType.getCiphertextSpace().getRing();
  lwe::CiphertextSpaceAttr ctAttr = lwe::CiphertextSpaceAttr::get(
      op.getContext(), ringAttr,
      ctType.getCiphertextSpace().getEncryptionType(), 2);
  lwe::LWECiphertextType outCTType = lwe::LWECiphertextType::get(
      op.getContext(), ctType.getApplicationData(), ctType.getPlaintextSpace(),
      ctAttr, ctType.getKey(), ctType.getModulusChain());
  KeySwitchInnerOp ksPoly =
      KeySwitchInnerOp::create(b, input2, op.getKeySwitchingKey());
  Value ksCT = lwe::FromCoeffsOp::create(
      b, outCTType, {ksPoly.getConstTerm(), ksPoly.getLinearTerm()});
  Value linearCT = lwe::FromCoeffsOp::create(b, outCTType, {input0, input1});
  Value sum = AddOp::create(b, ksCT, linearCT);
  rewriter.replaceOp(op, sum);
  return success();
}

struct DecomposeRelinearize
    : impl::DecomposeRelinearizeBase<DecomposeRelinearize> {
  using DecomposeRelinearizeBase::DecomposeRelinearizeBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeRelinearizePattern>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
