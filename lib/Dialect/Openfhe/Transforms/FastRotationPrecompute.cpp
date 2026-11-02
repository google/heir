#include "lib/Dialect/Openfhe/Transforms/FastRotationPrecompute.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"  // from @llvm-project

#define DEBUG_TYPE "fast-rotation-precompute"

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_FASTROTATIONPRECOMPUTE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

static LoopLikeOpInterface getOutermostLoopForInvariant(Value value,
                                                        Operation* op) {
  LoopLikeOpInterface outermostLoop = nullptr;
  Operation* current = op->getParentOp();
  while (current) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(current)) {
      if (loop.isDefinedOutsideOfLoop(value)) {
        outermostLoop = loop;
      } else {
        break;
      }
    }
    current = current->getParentOp();
  }
  return outermostLoop;
}

void processFunc(func::FuncOp funcOp, Value cryptoContext) {
  IRRewriter builder(funcOp->getContext());
  llvm::DenseMap<Value, llvm::SmallVector<RotOp>> ciphertextToRotateOps;
  llvm::DenseMap<Value, llvm::SmallDenseSet<int64_t>>
      ciphertextToDistinctRotations;
  llvm::DenseMap<Value, bool> ciphertextHasDynamicShift;
  llvm::DenseMap<Value, LoopLikeOpInterface> ciphertextToOutermostLoop;

  funcOp->walk([&](RotOp op) {
    Value ciphertext = op.getCiphertext();
    ciphertextToRotateOps[ciphertext].push_back(op);

    if (op.getStaticShift().has_value()) {
      ciphertextToDistinctRotations[ciphertext].insert(
          op.getStaticShift()->getValue().getZExtValue());
    } else if (op.getDynamicShift()) {
      if (auto constOp =
              op.getDynamicShift().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          ciphertextToDistinctRotations[ciphertext].insert(
              intAttr.getValue().getZExtValue());
        } else {
          ciphertextHasDynamicShift[ciphertext] = true;
        }
      } else {
        ciphertextHasDynamicShift[ciphertext] = true;
      }
    }

    if (LoopLikeOpInterface loop =
            getOutermostLoopForInvariant(ciphertext, op)) {
      auto it = ciphertextToOutermostLoop.find(ciphertext);
      if (it == ciphertextToOutermostLoop.end() || !it->second) {
        ciphertextToOutermostLoop[ciphertext] = loop;
      } else if (loop->isProperAncestor(it->second)) {
        ciphertextToOutermostLoop[ciphertext] = loop;
      }
    }
  });

  for (auto const& [ciphertext, rots] : ciphertextToRotateOps) {
    size_t distinctCount = 0;
    if (auto it = ciphertextToDistinctRotations.find(ciphertext);
        it != ciphertextToDistinctRotations.end()) {
      distinctCount = it->second.size();
    }
    bool hasDynamic = ciphertextHasDynamicShift.lookup(ciphertext);
    LoopLikeOpInterface outermostLoop =
        ciphertextToOutermostLoop.lookup(ciphertext);

    bool shouldPrecompute =
        (outermostLoop != nullptr) || (distinctCount >= 2) ||
        (hasDynamic && rots.size() >= 2) || (distinctCount >= 1 && hasDynamic);

    if (!shouldPrecompute) {
      continue;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Found ciphertext for fast rotation precomputation: "
               << ciphertext << "\n");

    // Insert the precomputation op right after the ciphertext is defined. If
    // the ciphertext is a block argument, the precomputation op is inserted at
    // the beginning of the block. Because the ciphertext is defined outside of
    // any enclosing loop where it is rotated, this insertion point dominates
    // all uses both inside and outside the loop.
    if (auto* definingOp = ciphertext.getDefiningOp()) {
      builder.setInsertionPointAfter(definingOp);
    } else {
      builder.setInsertionPointToStart(
          cast<BlockArgument>(ciphertext).getOwner());
    }

    auto precomputeOp = FastRotationPrecomputeOp::create(
        builder, ciphertext.getLoc(), cryptoContext, ciphertext);

    for (RotOp op : rots) {
      builder.setInsertionPoint(op);
      int cyclotomicOrder = 0;

      Value shiftValue;
      if (op.getStaticShift().has_value()) {
        int64_t rotationAmount = op.getStaticShift()->getValue().getSExtValue();
        shiftValue = arith::ConstantIndexOp::create(builder, op->getLoc(),
                                                    rotationAmount);
      } else if (op.getDynamicShift()) {
        shiftValue = op.getDynamicShift();
        if (!shiftValue.getType().isIndex()) {
          shiftValue = arith::IndexCastOp::create(
              builder, op->getLoc(), builder.getIndexType(), shiftValue);
        }
      } else {
        continue;
      }

      auto fastRot = FastRotationOp::create(
          builder, op->getLoc(), op.getType(), op.getCryptoContext(),
          op.getCiphertext(), shiftValue, builder.getIndexAttr(cyclotomicOrder),
          precomputeOp.getResult());
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
