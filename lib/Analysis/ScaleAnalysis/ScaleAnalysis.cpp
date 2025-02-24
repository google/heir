#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"

#include <functional>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "ScaleAnalysis"

namespace mlir::heir::polynomial {
llvm::APInt multiplicativeInverse(const llvm::APInt &x,
                                  const llvm::APInt &modulo);
}  // namespace mlir::heir::polynomial

namespace mlir {
namespace heir {

LogicalResult ScaleAnalysis::visitOperation(
    Operation *op, ArrayRef<const ScaleLattice *> operands,
    ArrayRef<ScaleLattice *> results) {
  auto t = schemeParam.getPlaintextModulus();
  auto qi = schemeParam.getQi();

  auto propagate = [&](Value value, const ScaleState &state) {
    auto *lattice = getLatticeElement(value);
    LLVM_DEBUG(llvm::dbgs()
               << "Propagate " << state << " to " << value << "\n");
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          // initialized to 1
          propagate(blockArg, ScaleState(1));
        }
      })
      .Case<arith::MulIOp>([&](auto mulOp) {
        // propagate scale to result
        auto lhs = operands[0]->getValue().getScale();
        auto rhs = operands[1]->getValue().getScale();
        auto result = lhs * rhs % t;
        propagate(mulOp.getResult(), ScaleState(result));
      })
      .Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        // propagate scale to result
        auto scale = operands[0]->getValue().getScale();
        // get level of the operand. MgmtAttr is attached to the result.
        auto level = getLevelFromMgmtAttr(modReduceOp) + 1;

        auto qInvT = ::mlir::heir::polynomial::multiplicativeInverse(
            APInt(64, qi[level] % t), APInt(64, t));

        auto newScale = scale * qInvT.getSExtValue() % t;

        propagate(modReduceOp.getResult(), ScaleState(newScale));
      })
      .Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        propagate(adjustScaleOp.getResult(),
                  ScaleState(adjustScaleOp.getScale()));
      })
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        // just propagate the scale
        auto scale = operands[0]->getValue().getScale();
        for (auto result : secretResults) {
          propagate(result, ScaleState(scale));
        }
      });
  return success();
}

void ScaleAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const ScaleLattice *> argumentLattices,
    ArrayRef<ScaleLattice *> resultLattices) {
  auto callback = std::bind(&ScaleAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<ScaleState, ScaleLattice>(
      call, argumentLattices, resultLattices, callback);
}

}  // namespace heir
}  // namespace mlir
