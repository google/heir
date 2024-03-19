#include "include/Analysis/TargetSlotAnalysis/TargetSlotAnalysis.h"

#include "lib/Dialect/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "target-slot-analysis"

namespace mlir {
namespace heir {
namespace target_slot_analysis {

void TargetSlotAnalysis::visitOperation(
    Operation *op, ArrayRef<TargetSlotLattice *> operands,
    ArrayRef<const TargetSlotLattice *> results) {
  llvm::TypeSwitch<Operation &>(*op)
      .Case<tensor::InsertOp>([&](auto insertOp) {
        LLVM_DEBUG({ llvm::dbgs() << "Visiting: " << *op << "\n"; });
        auto insertIndexRes = get1DExtractionIndex<tensor::InsertOp>(insertOp);
        // If the target slot can't be statically determined, we can't
        // propagate anything through the IR.
        if (failed(insertIndexRes)) return;

        // The target slot propagates to the value inserted, which is the first
        // positional argument
        TargetSlotLattice *lattice = operands[0];
        TargetSlot newSlot = TargetSlot{insertIndexRes.value()};
        LLVM_DEBUG({
          llvm::dbgs() << "Joining " << lattice->getValue() << " and "
                       << newSlot << " --> "
                       << TargetSlot::join(lattice->getValue(), newSlot)
                       << "\n";
        });
        ChangeResult changed = lattice->join(newSlot);
        propagateIfChanged(lattice, changed);
      })
      .Default([&](Operation &op) {
        // By default, an op propagates its result target slots to all its
        // operands.
        for (const TargetSlotLattice *r : results) {
          for (TargetSlotLattice *operand : operands) {
            ChangeResult result = operand->join(*r);
            propagateIfChanged(operand, result);
          }
        }
      });
}

}  // namespace target_slot_analysis
}  // namespace heir
}  // namespace mlir
