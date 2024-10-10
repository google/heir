#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"

#include <numeric>

#include "lib/Analysis/OptimizeRelinearizationAnalysis/OptimizeRelinearizationAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define DEBUG_TYPE "OptimizeRelinearization"

#define GEN_PASS_DEF_OPTIMIZERELINEARIZATION
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h.inc"

struct OptimizeRelinearization
    : impl::OptimizeRelinearizationBase<OptimizeRelinearization> {
  using OptimizeRelinearizationBase::OptimizeRelinearizationBase;

  // A helper to determine if the result type of an op needs to be fixed after
  // relinearizations are removed and re-inserted as a result of the
  // optimization.
  //
  // If it does not need patching, this function returns std::nullopt.
  // If it does need patching, this function returns an optional containing
  // the degree that the result type should have.
  std::optional<int> opNeedsResultTypePatching(Operation *op) {
    // The default case is to ensure all result RLWE ciphertexts
    // have the same degree as the input ciphertexts
    llvm::SmallVector<std::optional<int>, 4> resultDegrees;
    for (Value result : op->getResults()) {
      if (auto rlweType = dyn_cast<lwe::RLWECiphertextType>(result.getType())) {
        resultDegrees.push_back(rlweType.getRlweParams().getDimension() - 1);
        LLVM_DEBUG(llvm::dbgs() << "Result degree: "
                                << resultDegrees.back().value() << "\n");
      } else {
        resultDegrees.push_back(std::nullopt);
      }
    }

    llvm::SmallVector<std::optional<int>, 4> operandDegrees;
    for (Value operand : op->getOperands()) {
      if (auto rlweType =
              dyn_cast<lwe::RLWECiphertextType>(operand.getType())) {
        operandDegrees.push_back(rlweType.getRlweParams().getDimension() - 1);
        LLVM_DEBUG(llvm::dbgs() << "Operand degree: "
                                << operandDegrees.back().value() << "\n");
      } else {
        operandDegrees.push_back(std::nullopt);
      }
    }

    if (llvm::none_of(
            resultDegrees,
            [](std::optional<int> degree) { return degree.has_value(); }) ||
        llvm::none_of(operandDegrees, [](std::optional<int> degree) {
          return degree.has_value();
        })) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Either none of the operands or none of the results are "
                    "RLWE ciphertexts. There is nothing to do.\n");
      return std::nullopt;
    }

    int fixedDegree = operandDegrees.front().value();

    // Bail if any of the operand degrees disagrees from other operand
    // degrees.
    if (llvm::any_of(operandDegrees, [&](std::optional<int> degree) {
          return degree.has_value() && degree.value() != fixedDegree;
        })) {
      LLVM_DEBUG(llvm::dbgs() << "One or more operands have different "
                                 "key basis degrees.\n");
      return std::nullopt;
    }

    // Bail if the result degree is already the same as the operand
    // degrees
    if (llvm::all_of(resultDegrees, [&](std::optional<int> degree) {
          return degree.has_value() && degree.value() == fixedDegree;
        })) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Result key basis degree is already correct.\n");
      return std::nullopt;
    }

    return fixedDegree;
  }

  void processFunc(func::FuncOp funcOp) {
    MLIRContext *context = funcOp->getContext();

    // Remove all relin ops. This makes the IR invalid, because the key basis
    // sizes are incorrect. However, the correctness of the ILP ensures the key
    // basis sizes are made correct at the end.
    funcOp->walk([&](bgv::RelinearizeOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    OptimizeRelinearizationAnalysis analysis(funcOp);
    if (failed(analysis.solve())) {
      funcOp->emitError("Failed to solve the optimization problem");
      return signalPassFailure();
    }

    OpBuilder b(&getContext());

    funcOp->walk([&](Operation *op) {
      if (!analysis.shouldInsertRelin(op)) return;

      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting relin after: " << op->getName() << "\n");

      b.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        // Fill a vector with 0, 1, ..., degree
        int fromBasisDegree = 1 + analysis.keyBasisDegreeBeforeRelin(result);
        std::vector<int> range(fromBasisDegree);
        std::iota(std::begin(range), std::end(range), 0);

        DenseI32ArrayAttr beforeBasis = b.getDenseI32ArrayAttr(range);
        DenseI32ArrayAttr afterBasis = b.getDenseI32ArrayAttr({0, 1});
        auto reduceOp = b.create<bgv::RelinearizeOp>(op->getLoc(), result,
                                                     beforeBasis, afterBasis);
        result.replaceAllUsesExcept(reduceOp.getResult(), {reduceOp});
      }
    });

    // At this point we need to do some cleanup. The process of removing the
    // initial relinearize ops and inserting new ones did not update the key
    // basis sizes on result types. So we walk the IR and update them if
    // necessary.
    funcOp->walk([&](Operation *op) {
      TypeSwitch<Operation *, void>(op)
          .Case<bgv::RelinearizeOp>([&](auto op) {
            // correct by construction, nothing to do.
            return;
          })
          .Case<bgv::MulOp>([&](auto op) {
            auto lhsDeg =
                op.getLhs().getType().getRlweParams().getDimension() - 1;
            auto rhsDeg =
                op.getRhs().getType().getRlweParams().getDimension() - 1;
            auto resultDeg =
                op.getResult().getType().getRlweParams().getDimension() - 1;
            LLVM_DEBUG(llvm::dbgs()
                       << "Checking if key basis needs modifying in mul op: "
                       << op->getName() << "; lhsDeg=" << lhsDeg << "; rhsDeg="
                       << rhsDeg << "; resultDeg=" << resultDeg << "\n");

            if (lhsDeg + rhsDeg != resultDeg) {
              LLVM_DEBUG(llvm::dbgs() << "Updating result types for mul op\n");
              // Can't change the result type in place, so have to recreate
              // the operation from scratch.
              b.setInsertionPointAfter(op);
              lwe::RLWECiphertextType newResultType =
                  lwe::RLWECiphertextType::get(
                      context, op.getLhs().getType().getEncoding(),
                      lwe::RLWEParamsAttr::get(
                          context,
                          /*dimension=*/lhsDeg + rhsDeg,
                          /*ring=*/
                          op.getLhs().getType().getRlweParams().getRing()),
                      op.getLhs().getType().getUnderlyingType());
              auto newOp =
                  b.create<bgv::MulOp>(op->getLoc(), op.getLhs(), op.getRhs());
              op.getResult().replaceAllUsesWith(newOp.getResult());
              op.erase();
            }
          })
          .Default([&](Operation *op) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Checking if key basis needs modifying in: "
                       << op->getName() << "\n");

            if (op->getNumRegions() > 1) {
              LLVM_DEBUG(llvm::dbgs() << "Operation has regions. Skipping: "
                                      << op->getName() << "\n");
              return;
            }

            std::optional<int> fixedDegree = opNeedsResultTypePatching(op);
            if (fixedDegree == std::nullopt) return;

            LLVM_DEBUG(llvm::dbgs()
                       << "Fixing result type key basis degree for: "
                       << op->getName() << "; result degree being updated to "
                       << fixedDegree << "\n");

            // Construct the correct result types
            SmallVector<Type, 4> newResultTypes;
            for (Value result : op->getResults()) {
              if (auto rlweType =
                      dyn_cast<lwe::RLWECiphertextType>(result.getType())) {
                newResultTypes.push_back(lwe::RLWECiphertextType::get(
                    context, rlweType.getEncoding(),
                    lwe::RLWEParamsAttr::get(
                        context, /*dimension=*/1 + fixedDegree.value(),
                        rlweType.getRlweParams().getRing()),
                    rlweType.getUnderlyingType()));
              } else {
                newResultTypes.push_back(result.getType());
              }
            };

            // Replace the operation with the new result types
            Operation *newOp = cloneWithNewResultTypes(op, newResultTypes);
            b.setInsertionPointAfter(op);
            b.insert(newOp);

            op->getResults().replaceAllUsesWith(newOp->getResults());
            op->erase();
          });
    });
  }

  void runOnOperation() override {
    Operation *module = getOperation();
    module->walk([&](func::FuncOp op) { processFunc(op); });
  }
};

}  // namespace heir
}  // namespace mlir
