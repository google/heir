#include "lib/Dialect/Openfhe/Transforms/FastRotationPrecompute.h"

#include <cstdint>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

#define DEBUG_TYPE "fast-rotation-precompute"

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_FASTROTATIONPRECOMPUTE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

void processFunc(func::FuncOp funcOp, Value cryptoContext) {
  IRRewriter builder(funcOp->getContext());
  llvm::DenseMap<Value, llvm::SmallVector<RotOp>> ciphertextToRotateOps;
  llvm::DenseMap<Value, llvm::SmallDenseSet<int64_t>>
      ciphertextToDistinctRotations;
  funcOp->walk([&](RotOp op) {
    ciphertextToRotateOps[op.getCiphertext()].push_back(op);
    ciphertextToDistinctRotations[op.getCiphertext()].insert(
        op.getIndex().getValue().getZExtValue());
  });

  for (auto const& [ciphertext, rots] : ciphertextToDistinctRotations) {
    // TODO(#744): is there a meaningful tradeoff for fast precompute?
    if (rots.size() < 2) {
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "Found ciphertext with " << rots.size()
                            << " distinct rotations: " << ciphertext << "\n");

    // Insert the precomputation op right after the ciphertext is defined. If
    // the ciphertext is a block argument, the precomputation op is inserted at
    // the beginning of the block.
    if (auto* definingOp = ciphertext.getDefiningOp()) {
      builder.setInsertionPointAfter(definingOp);
    } else {
      builder.setInsertionPointToStart(
          cast<BlockArgument>(ciphertext).getOwner());
    }

    auto precomputeOp = FastRotationPrecomputeOp::create(
        builder, ciphertext.getLoc(), cryptoContext, ciphertext);

    for (RotOp op : ciphertextToRotateOps[ciphertext]) {
      builder.setInsertionPoint(op);
      // Cyclotomic order is 2*N where polynomial modulus is x^N + 1 This would
      // be the right value to use here, IF this actually ended up as the ring
      // dimension used by OpenFHE. However, OpenFHE sets its own parameters,
      // and so this ends up being ignored in favor of dynamically reading
      // `cc->GetRingDimension() * 2`.
      int cyclotomicOrder = 0;
      auto fastRot = FastRotationOp::create(
          builder, op->getLoc(), op.getType(), op.getCryptoContext(),
          op.getCiphertext(), op.getIndex(),
          builder.getIndexAttr(cyclotomicOrder), precomputeOp.getResult());
      builder.replaceOp(op, fastRot);
    }
  }
}

struct FastRotationPrecompute
    : impl::FastRotationPrecomputeBase<FastRotationPrecompute> {
  using FastRotationPrecomputeBase::FastRotationPrecomputeBase;

  void runOnOperation() override {
    // We must process funcs separately so that rotations are not attempted to
    // be batched across function boundaries.
    getOperation()->walk([&](func::FuncOp op) -> WalkResult {
      auto result = getArgOfType<openfhe::CryptoContextType>(op);
      if (failed(result)) {
        LDBG() << "Skipping func with no cryptocontext arg: " << op.getSymName()
               << "\n";
        return WalkResult::advance();
      }
      Value cryptoContext = result.value();
      processFunc(op, cryptoContext);
      return WalkResult::advance();
    });
  }
};
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
