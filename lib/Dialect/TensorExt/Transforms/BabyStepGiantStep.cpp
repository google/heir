#include "lib/Dialect/TensorExt/Transforms/BabyStepGiantStep.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <queue>
#include <set>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define DEBUG_TYPE "baby-step-giant-step"

#define GEN_PASS_DEF_BABYSTEPGIANTSTEP
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

struct RotationsAndMuls {
  std::set<int64_t> rotations;
  SmallVector<Operation *> muls;
};

template <typename MulOp>
struct SumOfRotatedProducts : public mlir::OpRewritePattern<arith::AddIOp> {
  SumOfRotatedProducts(Operation *top, DataFlowSolver *solver,
                       MLIRContext *context)
      : OpRewritePattern(context), top(top), solver(solver) {}

 public:
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // Match only on terminal add operations.
    // FIXME: this is actually too restrictive since the result can be added to
    // other things.
    if (llvm::any_of(op->getUsers(), [](auto user) {
          return llvm::isa<arith::AddIOp>(user);
        })) {
      return failure();
    }

    // Gather all summands.
    DenseMap<Value, bool> summands;
    std::queue<Operation *> addOps;
    addOps.push(op);
    for (; !addOps.empty(); addOps.pop()) {
      Operation *addOp = addOps.front();
      for (auto &operand : addOp->getOpOperands()) {
        auto definingAddOp =
            dyn_cast_or_null<arith::AddIOp>(operand.get().getDefiningOp());
        if (definingAddOp) {
          addOps.push(definingAddOp);
          continue;
        }
        summands[operand.get()] = true;
      }
    }
    if (summands.size() <= 2) return failure();

    // Gather all summands of the form ciphertext \arithop plaintext.
    SmallVector<Operation *> mulsToSum;
    SmallVector<int> rotationIndices;
    for (Value summand : summands.keys()) {
      // Summands must be defined from a multiplication operation.
      auto definingMulOp = dyn_cast_or_null<MulOp>(summand.getDefiningOp());
      if (!definingMulOp) {
        continue;
      }
      // One of the summands must be a plaintext value.
      auto plaintextIt = llvm::find_if(
          definingMulOp->getOpOperands(),
          [&](auto &operand) { return !isSecret(operand.get(), solver); });
      if (plaintextIt == definingMulOp->getOpOperands().end()) {
        continue;
      }

      // The other summand must be a secret, or a rotation of the secret.
      auto secretIt = llvm::find_if(
          definingMulOp->getOpOperands(),
          [&](auto &operand) { return isSecret(operand.get(), solver); });
      if (secretIt == definingMulOp->getOpOperands().end()) {
        continue;
      }

      mulsToSum.push_back(definingMulOp);
    }
    if (mulsToSum.size() <= 2) return failure();

    // Of all the summands that are ptxt * ctxt, we need to find those that
    // rotate a secret periodically.
    // TODO: probably also need to hold on to the mulOp reference
    DenseMap<Value, RotationsAndMuls> rotationsOfSecret;
    for (Operation *mulOp : mulsToSum) {
      // This exists
      auto secretOperand = llvm::find_if(
          mulOp->getOpOperands(),
          [&](auto &operand) { return isSecret(operand.get(), solver); });
      tensor_ext::RotateOp secretRotation =
          dyn_cast_or_null<RotateOp>(secretOperand->get().getDefiningOp());
      int64_t shift = 0;
      Value rotatedSecret;
      if (!secretRotation) {
        shift = 0;
        rotatedSecret = secretOperand->get();
      } else {
        // Only multiplication operations with constant shifts can be
        // considered.
        IntegerAttr shiftAttr;
        if (!matchPattern(secretRotation.getShift(), m_Constant(&shiftAttr))) {
          continue;
        }
        shift = static_cast<int64_t>(shiftAttr.getInt());
        rotatedSecret = secretRotation.getTensor();
      }
      if (!rotationsOfSecret.contains(rotatedSecret)) {
        rotationsOfSecret[rotatedSecret] = {{shift}, {mulOp}};
      } else {
        rotationsOfSecret[rotatedSecret].rotations.insert(shift);
        rotationsOfSecret[rotatedSecret].muls.push_back(mulOp);
      }
    }

    // Out of the rotated secrets, we need to find a set of periodic rotations.
    int64_t period = 0;
    Value secret;
    for (auto &[candidate, rotationsAndMuls] : rotationsOfSecret) {
      auto &rotations = rotationsAndMuls.rotations;
      if (rotations.size() <= 2) continue;
      auto it = rotations.begin();
      int prev = *it;
      // Not sure if this is necessary, but start at 0
      if (prev != 0) {
        op->emitRemark() << "Skipping secret with initial rotation of " << prev;
        continue;
      }
      std::advance(it, 1);
      period = *it - prev;
      while (it != rotations.end() && *it - period == prev) {
        prev = *it;
        std::advance(it, 1);
      }
      if (it != rotations.end()) {
        op->emitOpError()
            << "Failed to find a full periodic rotation of a secret.";
        continue;
      }
      LLVM_DEBUG(op->emitRemark() << "Found a periodic rotation of a secret.");
      // We found a periodic rotation of a secret!
      // FIXME: There may be extraneous muls of rotations of the secret, so
      // really we don't need to find an exact periodic rotation sequence. It
      // would suffice to find a satisfying subset.
      secret = candidate;
      break;
    }

    if (secret == nullptr) {
      return op->emitOpError()
             << "Failed to find a periodic rotation of a secret.";
    }

    // Rewrite the multiplication operations of the secret. Each rotation
    auto totalRotations = rotationsOfSecret[secret].rotations.size();
    // FIXME: handle non-square cases.
    if (!isPowerOfTwo(totalRotations)) {
      return failure();
    }

    rewriter.setInsertionPointToStart(op->getBlock());
    SmallVector<Value> babyStepRotations;
    auto numSteps = static_cast<int64_t>(std::floor(std::sqrt(totalRotations)));
    babyStepRotations.push_back(secret);
    for (int64_t i = 1; i < numSteps; ++i) {
      babyStepRotations.push_back(
          rewriter
              .create<tensor_ext::RotateOp>(
                  op->getLoc(), secret,
                  rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i))
              .getResult());
    }

    DenseMap<int64_t, SmallVector<Value>> groupsOfSummands;
    for (Operation *mulOp : rotationsOfSecret[secret].muls) {
      rewriter.setInsertionPointAfter(mulOp);
      // Get rotation amount
      auto secretOperand = llvm::find_if(
          mulOp->getOpOperands(),
          [&](auto &operand) { return isSecret(operand.get(), solver); });
      tensor_ext::RotateOp secretRotation =
          dyn_cast_or_null<RotateOp>(secretOperand->get().getDefiningOp());
      IntegerAttr shift = cast<IntegerAttr>(
          rewriter.getZeroAttr(IndexType::get(op->getContext())));
      Value tensor = secretOperand->get();
      if (secretRotation) {
        matchPattern(secretRotation.getShift(), m_Constant(&shift));
        tensor = secretRotation.getTensor();
      }
      int64_t shiftValue = static_cast<int64_t>(shift.getInt());
      auto babyStep = shiftValue % numSteps;
      auto remainder = shiftValue - babyStep;
      // Rewrite the multiplication: rot_{remainder} (
      // rot_{-remainder}(plaintext) * rot(babyStep)(ciphertext) )
      auto plaintextIt = llvm::find_if(
          mulOp->getOpOperands(),
          [&](auto &operand) { return !isSecret(operand.get(), solver); });
      auto inverseShift = rewriter.create<arith::ConstantOp>(
          op->getLoc(), IntegerAttr::get(shift.getType(), 0 - remainder));
      auto invRotatedPlaintext = rewriter.create<tensor_ext::RotateOp>(
          op->getLoc(), plaintextIt->get(), inverseShift);
      // TODO: set secretness of the new product and new inv rotation mul op
      auto newMulOp = rewriter.create<MulOp>(op->getLoc(), invRotatedPlaintext,
                                             babyStepRotations[babyStep]);
      auto invRotNewMulOp = rewriter.create<tensor_ext::RotateOp>(
          op->getLoc(), newMulOp,
          rewriter.create<arith::ConstantOp>(
              op->getLoc(), IntegerAttr::get(shift.getType(), remainder)));
      assert(summands.contains(mulOp->getResult(0)));
      summands.erase(mulOp->getResult(0));
      if (groupsOfSummands.contains(remainder)) {
        groupsOfSummands[remainder].push_back(invRotNewMulOp.getResult());
      } else {
        groupsOfSummands[remainder] = {invRotNewMulOp.getResult()};
      }
    }

    // Add all the groups of summands first together so that the rotation
    // optimizer can pass the rotation through the addition.
    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> newSummands;
    for (auto &[_, group] : groupsOfSummands) {
      Value newGroup = group[0];
      for (int i = 1; i < group.size(); ++i) {
        newGroup =
            rewriter.create<arith::AddIOp>(op->getLoc(), newGroup, group[i]);
      }
      newSummands.push_back(newGroup);
    }
    // newSummands have at least 2 elements in them.
    Value newSum = rewriter.create<arith::AddIOp>(op->getLoc(), newSummands[0],
                                                  newSummands[1]);
    for (int i = 2; i < newSummands.size(); i++) {
      newSum =
          rewriter.create<arith::AddIOp>(op->getLoc(), newSum, newSummands[i]);
    }
    for (auto summand : summands.keys()) {
      newSum = rewriter.create<arith::AddIOp>(op->getLoc(), newSum, summand);
    }

    // Replace the original add op.
    rewriter.replaceOp(op, newSum);
    return success();

    // FIXME Set secretness for all the new ops.

    // return solver->initializeAndRun(top);
    // return success();
  }

 private:
  // root operation the pass is on, should never be altered hence never null
  Operation *top;
  DataFlowSolver *solver;

  static inline void setValueToSecretness(DataFlowSolver *solver, Value value,
                                          Secretness secretness) {
    auto *lattice = solver->getOrCreateState<SecretnessLattice>(value);
    // solver->propagateIfChanged is bogus
    (void)lattice->join(secretness);
  }
};

struct BabyStepGiantStep : impl::BabyStepGiantStepBase<BabyStepGiantStep> {
  using BabyStepGiantStepBase::BabyStepGiantStepBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // First, apply canonicalizations to ensure that sequential rotations are
    // combined (e.g. rotate(rotate(v, 1), 1)) -> rotate(v, 2)
    RewritePatternSet preconditions(context);
    RotateOp::getCanonicalizationPatterns(preconditions, context);
    (void)applyPatternsGreedily(getOperation(), std::move(preconditions));

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    auto result = solver.initializeAndRun(getOperation());
    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // Second, check for baby-step giant-step pattern.
    RewritePatternSet patterns(context);
    patterns.add<SumOfRotatedProducts<arith::MulIOp>,
                 SumOfRotatedProducts<arith::MulFOp>>(getOperation(), &solver,
                                                      context);
    // Finally apply canonicalization patterns again to remove rotations by
    // factoring rotations through arithmetic ops
    populateFactorThroughPatterns(patterns, context);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));

    RewritePatternSet canonicalizations(context);
    RotateOp::getCanonicalizationPatterns(canonicalizations, context);
    (void)applyPatternsGreedily(getOperation(), std::move(canonicalizations));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
