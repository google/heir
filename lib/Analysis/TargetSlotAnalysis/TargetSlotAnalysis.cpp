#include "lib/Analysis/TargetSlotAnalysis/TargetSlotAnalysis.h"

#include "lib/Dialect/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
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

LogicalResult TargetSlotAnalysis::visitOperation(
    Operation *op, ArrayRef<TargetSlotLattice *> operands,
    ArrayRef<const TargetSlotLattice *> results) {
  llvm::TypeSwitch<Operation &>(*op)
      .Case<tensor::InsertOp>([&](auto insertOp) {
        LLVM_DEBUG({ llvm::dbgs() << "Visiting: " << *op << "\n"; });
        auto insertIndices = insertOp.getIndices();
        if (insertIndices.size() != 1) {
          LLVM_DEBUG(llvm::dbgs() << "At " << insertOp
                                  << " can't handle >1D insertion index\n");
          return;
        }

        Value insertIndexValue = insertOp.getIndices()[0];
        const dataflow::Lattice<dataflow::ConstantValue> *insertIndexLattice =
            sccpAnalysis
                ->lookupState<dataflow::Lattice<dataflow::ConstantValue>>(
                    insertIndexValue);

        if (insertIndexLattice) {
          LLVM_DEBUG(llvm::dbgs()
                     << "At " << insertOp << " SCCP analysis gives lattice of "
                     << *insertIndexLattice << "\n");
        }

        // If the target slot can't be statically determined, we can't
        // propagate anything through the IR.
        if (!insertIndexLattice ||
            insertIndexLattice->getValue().isUninitialized() ||
            !insertIndexLattice->getValue().getConstantValue()) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "At " << insertOp
              << " can't statically determine constant insertion index\n");
          return;
        }
        Attribute insertIndexAttr =
            insertIndexLattice->getValue().getConstantValue();
        auto insertIndexIntAttr = mlir::dyn_cast<IntegerAttr>(insertIndexAttr);
        assert(insertIndexIntAttr &&
               "If 1D insertion index is constant, it must be integer");
        int64_t insertIndexConst = insertIndexIntAttr.getInt();

        // The target slot propagates to the value inserted, which is the first
        // positional argument
        TargetSlotLattice *lattice = operands[0];
        TargetSlot newSlot = TargetSlot{insertIndexConst};
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
  return mlir::success();
}

}  // namespace target_slot_analysis
}  // namespace heir
}  // namespace mlir
