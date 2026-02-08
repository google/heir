#include "lib/Dialect/Openfhe/Transforms/ConvertToExtendedBasis.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_CONVERTTOEXTENDEDBASIS
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct ConvertToExtendedBasis
    : impl::ConvertToExtendedBasisBase<ConvertToExtendedBasis> {
  using ConvertToExtendedBasisBase::ConvertToExtendedBasisBase;

  void runOnOperation() override {
    IRRewriter builder(getOperation()->getContext());

    SmallVector<FastRotationOp> fastRotOps;
    getOperation()->walk([&](FastRotationOp op) { fastRotOps.push_back(op); });

    for (FastRotationOp op : fastRotOps) {
      builder.setInsertionPoint(op);

      // Create FastRotationExtOp - first rotation should have addFirst=true
      // For now, we set addFirst=true for all; the hoisting pass will optimize
      auto fastRotExt = FastRotationExtOp::create(
          builder, op->getLoc(), op.getType(), op.getCryptoContext(),
          op.getInput(), op.getIndex(), op.getPrecomputedDigitDecomp(),
          /*addFirst=*/true);

      // Create KeySwitchDownOp to convert back from extended basis
      auto keySwitchDown = KeySwitchDownOp::create(
          builder, op->getLoc(), op.getType(), op.getCryptoContext(),
          fastRotExt.getResult());

      op.replaceAllUsesWith(keySwitchDown.getResult());
      op.erase();
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
