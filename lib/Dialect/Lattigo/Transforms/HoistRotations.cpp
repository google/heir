#include "lib/Dialect/Lattigo/Transforms/HoistRotations.h"

#include <cstdint>
#include <memory>

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

#define DEBUG_TYPE "lattigo-hoist-rotations"

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_DEF_LATTIGOHOISTROTATIONS
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

struct LattigoHoistRotations
    : impl::LattigoHoistRotationsBase<LattigoHoistRotations> {
  using LattigoHoistRotationsBase::LattigoHoistRotationsBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp op) -> WalkResult {
      // CKKS
      auto ckksResult = getArgOfType<lattigo::CKKSEvaluatorType>(op);
      if (succeeded(ckksResult)) {
        Value evaluator = ckksResult.value();
        processFunc<lattigo::CKKSRotateNewOp, lattigo::CKKSRotateHoistedNewOp,
                    lattigo::RLWELookupRotatedOp>(op, evaluator);
        return WalkResult::advance();
      }

      // TODO(#2214): Add support for BGV/BFV. This requires handling Lattigo
      // BGV RotateColumns ops and defining corresponding BGV hoisted ops.

      LDBG() << "Skipping func with no lattigo evaluator arg: "
             << op.getSymName() << "\n";
      return WalkResult::advance();
    });
  }

 private:
  template <typename RotateOp, typename HoistedRotateOp, typename LookupOp>
  void processFunc(func::FuncOp funcOp, Value evaluator) {
    IRRewriter builder(funcOp->getContext());
    llvm::DenseMap<Value, llvm::SmallVector<RotateOp>> ciphertextToRotateOps;
    llvm::DenseMap<Value, llvm::DenseSet<int64_t>>
        ciphertextToDistinctRotations;
    llvm::SmallVector<Value> ciphertexts;

    funcOp->walk([&](RotateOp op) {
      int64_t shift = 0;
      bool hasConstantShift = false;
      if (op.getStaticShiftAttr()) {
        shift = op.getStaticShiftAttr().getInt();
        hasConstantShift = true;
      } else if (op.getDynamicShift()) {
        Attribute attr;
        if (matchPattern(op.getDynamicShift(), m_Constant(&attr))) {
          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
            shift = intAttr.getInt();
            hasConstantShift = true;
          }
        }
      }

      if (hasConstantShift) {
        if (!ciphertextToRotateOps.count(op.getInput())) {
          ciphertexts.push_back(op.getInput());
        }
        ciphertextToRotateOps[op.getInput()].push_back(op);
        ciphertextToDistinctRotations[op.getInput()].insert(shift);
      }
    });

    for (Value ciphertext : ciphertexts) {
      auto const& rots = ciphertextToDistinctRotations[ciphertext];
      if (rots.size() < 2) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "Found ciphertext with " << rots.size()
                              << " distinct rotations: " << ciphertext << "\n");

      if (auto* definingOp = ciphertext.getDefiningOp()) {
        builder.setInsertionPointAfter(definingOp);
      } else {
        builder.setInsertionPointToStart(
            cast<BlockArgument>(ciphertext).getOwner());
      }

      SmallVector<int64_t> offsets;
      for (int64_t rot : rots) {
        offsets.push_back(rot);
      }
      llvm::sort(offsets);

      auto hoistedRotateOp = HoistedRotateOp::create(
          builder, ciphertext.getLoc(),
          lattigo::RLWERotatedCiphertextListType::get(builder.getContext()),
          evaluator, ciphertext, builder.getDenseI64ArrayAttr(offsets));

      for (RotateOp op : ciphertextToRotateOps[ciphertext]) {
        builder.setInsertionPoint(op);
        IntegerAttr shiftAttr = op.getStaticShiftAttr();
        if (!shiftAttr) {
          Attribute attr;
          matchPattern(op.getDynamicShift(), m_Constant(&attr));
          shiftAttr = cast<IntegerAttr>(attr);
        }
        auto lookupOp =
            LookupOp::create(builder, op.getLoc(), op.getType(),
                             hoistedRotateOp.getResult(), shiftAttr);
        builder.replaceOp(op, lookupOp.getResult());
      }
    }
  }
};

std::unique_ptr<Pass> createLattigoHoistRotations() {
  return std::make_unique<LattigoHoistRotations>();
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
