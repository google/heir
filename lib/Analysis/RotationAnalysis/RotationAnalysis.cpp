#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace rotation_analysis {

void RotationAnalysis::run(Operation *op) {
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // If the op has no tensor results and no regions, then there's nothing to
    // do. The operation may consume a tensor but cannot further reduce it.
    if (op->getNumRegions() == 0 &&
        llvm::none_of(op->getResultTypes(), [](Type type) {
          return mlir::isa<RankedTensorType>(type);
        })) {
      return WalkResult::advance();
    }

    // Each tensor result can be the start of a new reduction.
    for (Value result : op->getResults()) {
      initializeFromValueIfTensor(result);
    }

    // Block args within regions can be the start of a new reduction.
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Value arg : block.getArguments()) {
          initializeFromValueIfTensor(arg);
        }
      }
    }

    // Each op now gets special treatment.
    //
    // - Rotate ops shift the accessIndices of their tensor operand's
    //   reductions if the shift is known to be constant.
    // - Binary ops join partial reductions of operands and set the opName.
    // - Everything else is ignored.
    llvm::TypeSwitch<Operation &>(*op)
        .Case<tensor_ext::RotateOp>([&](auto rotateOp) {
          LLVM_DEBUG({ llvm::dbgs() << "Visiting: " << *op << "\n"; });
          const dataflow::Lattice<dataflow::ConstantValue> *shiftLattice =
              solver.lookupState<dataflow::Lattice<dataflow::ConstantValue>>(
                  rotateOp.getShift());

          if (shiftLattice) {
            LLVM_DEBUG(llvm::dbgs() << "At " << rotateOp
                                    << " SCCP analysis gives lattice of "
                                    << *shiftLattice << "\n");
          }

          // If the rotation shift can't be statically determined, we can't
          // propagate anything through the IR.
          if (!shiftLattice || shiftLattice->getValue().isUninitialized() ||
              !shiftLattice->getValue().getConstantValue()) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "At " << rotateOp
                << " can't statically determine constant insertion index\n");
            return;
          }
          auto shiftValue = mlir::dyn_cast<IntegerAttr>(
                                shiftLattice->getValue().getConstantValue())
                                .getInt();

          // For each partial reduction the tensor operand is a root of,
          // rotate the accessed indices appropriately.
          Value tensor = rotateOp.getTensor();
          Value result = rotateOp.getResult();
          for (const auto &reduction : rootToPartialReductions[tensor]) {
            addPartialReduction(
                PartialReduction::rotate(reduction, shiftValue, result));
          }
        })
        .Case<arith::AddIOp, arith::MulIOp>([&](auto arithOp) {
          LLVM_DEBUG({ llvm::dbgs() << "Visiting: " << arithOp << "\n"; });
          Value lhs = arithOp.getLhs();
          Value rhs = arithOp.getRhs();
          Value newRoot = arithOp.getResult();
          OperationName opName = arithOp.getOperation()->getName();

          // This is inefficient, but what can we do better here? I suspect a
          // better approach may be to identify cases in which only one of these
          // reductions needs to be kept because it's "the best" according to
          // some metric (e.g., it monotonically increases the number of indices
          // and all else stays the same). But for now even on the
          // box_blur_64x64 example this is far from the bottleneck.
          for (const auto &lhsReduction : rootToPartialReductions[lhs]) {
            for (const auto &rhsReduction : rootToPartialReductions[rhs]) {
              if (PartialReduction::canJoin(lhsReduction, rhsReduction,
                                            opName)) {
                addPartialReduction(PartialReduction::join(
                    lhsReduction, rhsReduction, newRoot, opName));
              }
            }
          }
        });

    return WalkResult::advance();
  });
}

}  // namespace rotation_analysis
}  // namespace heir
}  // namespace mlir
