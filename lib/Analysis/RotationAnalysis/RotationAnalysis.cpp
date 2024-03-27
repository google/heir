#include "include/Analysis/RotationAnalysis/RotationAnalysis.h"

#include "include/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace rotation_analysis {

void RotationAnalysis::visitOperation(
    Operation *op, ArrayRef<const RotationLattice *> operands,
    ArrayRef<RotationLattice *> results) {
  llvm::TypeSwitch<Operation &>(*op)
      .Case<tensor_ext::RotateOp>([&](auto rotateOp) {
        LLVM_DEBUG({ llvm::dbgs() << "Visiting: " << *op << "\n"; });
        auto shiftConstantOp =
            rotateOp.getShift().template getDefiningOp<arith::ConstantOp>();
        // If the rotation shift can't be statically determined, we can't
        // propagate anything through the IR.
        if (!shiftConstantOp) return;

        int64_t shiftValue =
            dyn_cast<IntegerAttr>(shiftConstantOp.getValue()).getInt();

        // The target slot propagates from the tensor argument to the result;
        // the tensor argument is first in the tablegen definition.
        const RotationLattice *lattice = operands[0];
        RotationSets latticeRotations = lattice->getValue();

        // If it's a block argument, then there is no initialized lattice value
        // and we can override it with a "zero rotation"
        auto blockArg = dyn_cast<BlockArgument>(rotateOp.getTensor());
        if (blockArg) {
          latticeRotations = RotationSets::from(blockArg);
        }
        RotationSets rotated =
            RotationSets::rotate(latticeRotations, shiftValue);

        for (RotationLattice *r : results) {
          ChangeResult result = r->join(rotated);
          propagateIfChanged(r, result);
        }
      })
      .Default([&](Operation &op) {
        // By default, an op propagates its result target slots to all its
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto *latticeOperand = operands[operand.getOperandNumber()];

          for (RotationLattice *r : results) {
            ChangeResult result = r->join(*latticeOperand);
            // If the operand is a block arg, this additionally treats this as
            // a zero rotation. If the underlying tensor differs across
            // operands, this will also cause a Status::TooManyTensors.
            // Otherwise, the join is a no-op.
            result |= r->join(RotationSets::from(operand.get()));
            propagateIfChanged(r, result);
          }
        }
      });
}

void RotationAnalysis::setToEntryState(RotationLattice *lattice) {
  lattice->getValue().clear();
}

}  // namespace rotation_analysis
}  // namespace heir
}  // namespace mlir
