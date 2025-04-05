#include "lib/Analysis/AddAndKeySwitchCountAnalysis/AddAndKeySwitchCountAnalysis.h"

#include <algorithm>
#include <tuple>

#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "AddAndKeySwitchCountAnalysis"

namespace mlir {
namespace heir {

LogicalResult CountAnalysis::visitOperation(
    Operation *op, ArrayRef<const CountLattice *> operands,
    ArrayRef<CountLattice *> results) {
  auto propagate = [&](Value value, const CountState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          // one Vfresh
          propagate(blockArg, CountState(1, 0));
        }
      })
      .Case<arith::AddIOp, arith::SubIOp, arith::AddFOp, arith::SubFOp>(
          [&](auto &op) {
            // condition on result secretness
            SmallVector<OpResult> secretResults;
            getSecretResults(op, secretResults);
            if (secretResults.empty()) {
              return;
            }

            CountState zeroState(0, 0);
            SmallVector<OpOperand *> secretOperands;
            getSecretOperands(op, secretOperands);
            for (auto *operand : secretOperands) {
              auto countState =
                  operands[operand->getOperandNumber()]->getValue();
              zeroState = zeroState + countState;
            }

            for (auto result : secretResults) {
              propagate(result, zeroState);
            }
          })
      .Case<arith::MulIOp, arith::MulFOp>([&](auto &op) {
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        // now noise is Vmult
        // TODO(#1168): we can actually do a more fine grained analysis here
        // distinguishing ct-ct and ct-pt
        propagate(op.getResult(), CountState(1, 0));
      })
      .Case<mgmt::RelinearizeOp, tensor_ext::RotateOp>([&](auto &op) {
        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(op, secretOperands);
        if (secretOperands.empty()) {
          return;
        }

        auto state = operands[0]->getValue();
        if (!state.isInitialized()) {
          return;
        }

        propagate(op.getResult(), state.keySwitch());
      })
      // TODO(#1174): in BGV tensor::ExtractOp is assumed to be always
      // mul+const
      .Case<tensor::ExtractOp>([&](auto &op) {
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        // now noise is Vmult + one Vks
        propagate(op.getResult(), CountState(1, 1));
      })
      .Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        // implicitly ensure that the operand is secret

        // should not propagate through mgmt::ModReduceOp, reset
        propagate(modReduceOp.getResult(), CountState(0, 0));
      })
      .Default(
          [&](auto &op) {
            // condition on result secretness
            SmallVector<OpResult> secretResults;
            getSecretResults(&op, secretResults);
            if (secretResults.empty()) {
              return;
            }

            if (!mlir::isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                           arith::ExtFOp, mgmt::InitOp>(op)) {
              op.emitError()
                  << "Unsupported operation for count analysis encountered.";
            }

            SmallVector<OpOperand *> secretOperands;
            getSecretOperands(&op, secretOperands);
            if (secretOperands.empty()) {
              return;
            }

            // inherit count from the first secret operand
            CountState first;
            for (auto *operand : secretOperands) {
              auto &countState = getLatticeElement(operand->get())->getValue();
              if (!countState.isInitialized()) {
                return;
              }
              first = countState;
              break;
            }

            for (auto result : secretResults) {
              propagate(result, first);
            }
          });
  return success();
}

void annotateCount(Operation *top, DataFlowSolver *solver) {
  [[maybe_unused]] auto getIntegerAttr = [&](int level) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), level);
  };

  auto maxAddCount = 0;
  auto maxKeySwitchCount = 0;

  auto getCount = [&](Value value) {
    auto state = solver->lookupState<CountLattice>(value)->getValue();
    // update the max
    maxAddCount = std::max(maxAddCount, state.getAddCount());
    maxKeySwitchCount = std::max(maxKeySwitchCount, state.getKeySwitchCount());
    return std::make_tuple(state.getAddCount(), state.getKeySwitchCount());
  };

  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    for (auto i = 0; i != genericOp.getBody()->getNumArguments(); ++i) {
      auto blockArg = genericOp.getBody()->getArgument(i);
      [[maybe_unused]] auto [addCount, keySwitchCount] = getCount(blockArg);
      // only annotate each Value when debugging
      LLVM_DEBUG({
        if (addCount != 0) {
          genericOp.setOperandAttr(i, "addCount", getIntegerAttr(addCount));
        }
        if (keySwitchCount != 0) {
          genericOp.setOperandAttr(i, "keySwitchCount",
                                   getIntegerAttr(keySwitchCount));
        }
      });
    }

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      [[maybe_unused]] auto [addCount, keySwitchCount] =
          getCount(op->getResult(0));
      // only annotate each Value when debugging
      LLVM_DEBUG({
        if (addCount != 0) {
          op->setAttr("openfhe.addCount", getIntegerAttr(addCount));
        }
        if (keySwitchCount != 0) {
          op->setAttr("openfhe.keySwitchCount", getIntegerAttr(keySwitchCount));
        }
      });
    });

    // annotate mgmt::OpenfheParamsAttr to func::FuncOp containing the genericOp
    auto *funcOp = genericOp->getParentOp();
    auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
        funcOp->getContext(), maxAddCount, maxKeySwitchCount);
    funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName,
                    openfheParamAttr);
  });
}

}  // namespace heir
}  // namespace mlir
