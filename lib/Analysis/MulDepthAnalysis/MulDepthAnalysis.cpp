#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"

#include <functional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult MulDepthAnalysis::visitOperation(
    Operation *op, ArrayRef<const MulDepthLattice *> operands,
    ArrayRef<MulDepthLattice *> results) {
  auto propagate = [&](Value value, const MulDepthState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, MulDepthState(0));
        }
      })
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        auto isMul = false;

        if (isa<arith::MulIOp, arith::MulFOp, mgmt::AdjustScaleOp>(op)) {
          isMul = true;
        }

        // NOTE: special case for ExtractOp... it is a mulconst+rotate
        // if not annotated with slot_extract
        // TODO(#1174): decide packing earlier in the pipeline instead of
        // annotation
        if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
          if (!extractOp->getAttr("slot_extract")) {
            // must be true
            isMul = true;
          }
        }

        // inherit mul result from secret operands
        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(&op, secretOperands);
        int operandsMulDepth = 0;
        for (auto *operand : secretOperands) {
          auto &mulResultState = getLatticeElement(operand->get())->getValue();
          if (!mulResultState.isInitialized()) {
            return;
          }
          operandsMulDepth =
              std::max(operandsMulDepth, mulResultState.getMulDepth());
        }

        auto resultMulDepth = operandsMulDepth + (isMul ? 1 : 0);

        for (auto result : secretResults) {
          propagate(result, MulDepthState(resultMulDepth));
        }
      });
  return success();
}

void MulDepthAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const MulDepthLattice *> argumentLattices,
    ArrayRef<MulDepthLattice *> resultLattices) {
  auto callback = std::bind(&MulDepthAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<MulDepthState, MulDepthLattice>(
      call, argumentLattices, resultLattices, callback);
}

}  // namespace heir
}  // namespace mlir
