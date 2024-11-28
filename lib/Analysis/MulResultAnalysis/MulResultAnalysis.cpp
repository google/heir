#include "lib/Analysis/MulResultAnalysis/MulResultAnalysis.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
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

  auto ensureSecretness = [&](Operation *op, Value value) -> bool {
    // create dependency on SecretnessAnalysis
    auto *lattice =
        getOrCreateFor<SecretnessLattice>(getProgramPointAfter(op), value);
    if (!lattice->getValue().isInitialized()) {
      return false;
    }
    return lattice->getValue().getSecretness();
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
        if (op.getNumResults() == 0) {
          return;
        }

        // condition on result secretness
        auto secretness = ensureSecretness(&op, op.getResult(0));
        if (!secretness) {
          return;
        }

        auto isMulResult = false;
        // NOTE: special case for ExtractOp... it is a mulconst+rotate
        if (isa<arith::MulIOp, arith::MulFOp, tensor::ExtractOp>(op)) {
          isMulResult = true;
        }

        for (const auto *operand : operands) {
          auto secretness = ensureSecretness(&op, operand->getAnchor());
          if (!secretness) {
            continue;
          }
          // now operand is secret
          if (!operand->getValue().isInitialized()) {
            return;
          }
          isMulResult = isMulResult || operand->getValue().getIsMulResult();
        }

        for (auto result : op.getResults()) {
          propagate(result, MulResultState(isMulResult));
        }
      });
  return success();
}

}  // namespace heir
}  // namespace mlir
