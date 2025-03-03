#include "lib/Analysis/MulResultAnalysis/MulResultAnalysis.h"

#include <functional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult MulResultAnalysis::visitOperation(
    Operation *op, ArrayRef<const MulResultLattice *> operands,
    ArrayRef<MulResultLattice *> results) {
  auto propagate = [&](Value value, const MulResultState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, MulResultState(false));
        }
      })
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        auto isMulResult = false;

        if (isa<arith::MulIOp, arith::MulFOp>(op)) {
          isMulResult = true;
        }

        // NOTE: special case for ExtractOp... it is a mulconst+rotate
        // if not annotated with slot_extract
        // TODO(#1174): decide packing earlier in the pipeline instead of
        // annotation
        if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
          if (!extractOp->getAttr("slot_extract")) {
            // must be true
            isMulResult = true;
          }
        }

        // inherit mul result from secret operands
        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(&op, secretOperands);
        for (auto *operand : secretOperands) {
          auto &mulResultState = getLatticeElement(operand->get())->getValue();
          if (!mulResultState.isInitialized()) {
            return;
          }
          isMulResult = isMulResult || mulResultState.getIsMulResult();
        }

        for (auto result : secretResults) {
          propagate(result, MulResultState(isMulResult));
        }
      });
  return success();
}

void MulResultAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const MulResultLattice *> argumentLattices,
    ArrayRef<MulResultLattice *> resultLattices) {
  auto callback = std::bind(&MulResultAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<MulResultState, MulResultLattice>(
      call, argumentLattices, resultLattices, callback);
}

}  // namespace heir
}  // namespace mlir
