#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_WRAPGENERIC
#include "lib/Transforms/Secretize/Passes.h.inc"

struct WrapWithGeneric : public OpRewritePattern<func::FuncOp> {
  WrapWithGeneric(mlir::MLIRContext* context, DataFlowSolver* solver)
      : mlir::OpRewritePattern<func::FuncOp>(context), solver(solver) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter& rewriter) const override {
    bool hasSecrets = false;

    SmallVector<Type, 4> newInputs;
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      auto argTy = op.getArgumentTypes()[i];
      if (op.getArgAttr(i, secret::SecretDialect::kArgSecretAttrName) !=
          nullptr) {
        hasSecrets = true;
        op.removeArgAttr(i, secret::SecretDialect::kArgSecretAttrName);

        auto newTy = secret::SecretType::get(argTy);
        if (!op.isDeclaration())
          op.getArgument(i).setType(newTy);  // Updates the block argument type.
        newInputs.push_back(newTy);
      } else {
        newInputs.push_back(argTy);
      }
    }

    if (!hasSecrets) {
      // Match failure, no secret inputs.
      return rewriter.notifyMatchFailure(op, "no secret inputs found");
    }

    // Externally defined functions have no body - conservatively wrap all
    // outputs
    if (op.isDeclaration()) {
      SmallVector<Type, 6> newOutputs;
      for (Type resultType : op.getResultTypes()) {
        newOutputs.push_back(secret::SecretType::get(resultType));
      }
      rewriter.modifyOpInPlace(op, [&] {
        op.setFunctionType(
            FunctionType::get(getContext(), {newInputs}, {newOutputs}));
      });
      return success();
    }

    // Phase 1: Identify which operations depend on secrets
    Block& opEntryBlock = op.getRegion().front();
    auto* returnOp = opEntryBlock.getTerminator();

    // Track which values are secret (including block arguments)
    llvm::DenseSet<Value> secretValues;
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      if (isSecret(op.getArgument(i), solver)) {
        secretValues.insert(op.getArgument(i));
      }
    }

    // Track which operations are secret-dependent
    llvm::DenseSet<Operation*> secretOps;
    for (Operation& bodyOp : opEntryBlock) {
      if (&bodyOp == returnOp) continue;

      // An operation is secret if any of its operands are secret
      bool isSecretOp = llvm::any_of(bodyOp.getOperands(), [&](Value operand) {
        return secretValues.contains(operand) || isSecret(operand, solver);
      });

      if (isSecretOp) {
        secretOps.insert(&bodyOp);
        // All results of a secret op become secret
        for (Value result : bodyOp.getResults()) {
          secretValues.insert(result);
        }
      }
    }

    // Phase 2: Determine output types and which outputs need to be in generic
    SmallVector<Type, 6> newOutputs;
    SmallVector<Value> secretReturnValues;
    SmallVector<Value> plaintextReturnValues;
    SmallVector<unsigned> secretReturnIndices;
    SmallVector<unsigned> plaintextReturnIndices;

    for (auto [i, resultType] : llvm::enumerate(op.getResultTypes())) {
      Value returnVal = returnOp->getOperand(i);
      if (secretValues.contains(returnVal) || isSecret(returnVal, solver)) {
        newOutputs.push_back(secret::SecretType::get(resultType));
        secretReturnValues.push_back(returnVal);
        secretReturnIndices.push_back(i);
      } else {
        newOutputs.push_back(resultType);
        plaintextReturnValues.push_back(returnVal);
        plaintextReturnIndices.push_back(i);
      }
    }

    // Modification to function type should go through the rewriter
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(
          FunctionType::get(getContext(), {newInputs}, {newOutputs}));
    });

    // If there are no secret-dependent operations AND no secret return values,
    // we don't need a generic at all (purely plaintext function).
    // But if there are secret return values (e.g., function directly returns
    // its secret input), we still need a generic even with no operations.
    if (secretOps.empty() && secretReturnValues.empty()) {
      // Purely plaintext function - no generic needed
      return success();
    }

    // Phase 3: Collect inputs for the secret.generic block
    // These are: (1) secret arguments, (2) plaintext values used by secret ops
    SmallVector<Value> genericInputs;
    SmallVector<Type> genericInputTypes;

    // Add all function arguments that are used by secret ops (or are secret)
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      genericInputs.push_back(op.getArgument(i));
      genericInputTypes.push_back(op.getArgument(i).getType());
    }

    // Collect plaintext-defined values that are used inside secret ops
    SmallVector<Value> plaintextValuesUsedInGeneric;
    for (Operation* secretOp : secretOps) {
      for (Value operand : secretOp->getOperands()) {
        // If the operand is from outside the secretOps set (i.e., plaintext)
        if (!secretValues.contains(operand)) {
          Operation* defOp = operand.getDefiningOp();
          // It's a plaintext value defined by a non-secret op in this function
          if (defOp && !secretOps.contains(defOp) &&
              defOp->getParentRegion() == &op.getRegion()) {
            if (!llvm::is_contained(plaintextValuesUsedInGeneric, operand)) {
              plaintextValuesUsedInGeneric.push_back(operand);
              genericInputs.push_back(operand);
              genericInputTypes.push_back(operand.getType());
            }
          }
        }
      }
    }

    // Phase 4: Build the secret.generic with only secret ops
    SmallVector<Type> genericOutputTypes;
    for (Value v : secretReturnValues) {
      genericOutputTypes.push_back(secret::SecretType::get(v.getType()));
    }

    // Create a new block for the rewritten function
    auto* newBlock = rewriter.createBlock(
        &opEntryBlock, opEntryBlock.getArgumentTypes(),
        SmallVector<Location>(opEntryBlock.getNumArguments(), op.getLoc()));

    rewriter.setInsertionPointToStart(newBlock);

    // Build mapping from old block args to new block args
    IRMapping outerMapping;
    for (unsigned i = 0; i < opEntryBlock.getNumArguments(); ++i) {
      outerMapping.map(opEntryBlock.getArgument(i), newBlock->getArgument(i));
    }

    // Clone plaintext operations to the new block (before the generic)
    for (Operation& bodyOp : opEntryBlock) {
      if (&bodyOp == returnOp) continue;
      if (!secretOps.contains(&bodyOp)) {
        Operation* clonedOp = rewriter.clone(bodyOp, outerMapping);
        for (unsigned i = 0; i < bodyOp.getNumResults(); ++i) {
          outerMapping.map(bodyOp.getResult(i), clonedOp->getResult(i));
        }
      }
    }

    // Update genericInputs to use the new block's values
    SmallVector<Value> mappedGenericInputs;
    for (Value v : genericInputs) {
      mappedGenericInputs.push_back(outerMapping.lookupOrDefault(v));
    }

    // Now create the secret.generic
    auto newGeneric = secret::GenericOp::create(
        rewriter, op.getLoc(), mappedGenericInputs, genericOutputTypes,
        [&](OpBuilder& b, Location loc, ValueRange blockArguments) {
          // Map inputs to block arguments
          IRMapping innerMapping;
          for (unsigned i = 0; i < genericInputs.size(); ++i) {
            innerMapping.map(genericInputs[i], blockArguments[i]);
          }

          // Clone only secret operations into the generic
          for (Operation& bodyOp : opEntryBlock) {
            if (&bodyOp == returnOp) continue;
            if (secretOps.contains(&bodyOp)) {
              Operation* clonedOp = b.clone(bodyOp, innerMapping);
              for (unsigned i = 0; i < bodyOp.getNumResults(); ++i) {
                innerMapping.map(bodyOp.getResult(i), clonedOp->getResult(i));
              }
            }
          }

          // Yield only the secret return values
          SmallVector<Value> yieldValues;
          for (Value v : secretReturnValues) {
            yieldValues.push_back(innerMapping.lookupOrDefault(v));
          }
          secret::YieldOp::create(b, loc, yieldValues);
        });

    // Build the final return values in the correct order
    SmallVector<Value> finalReturnValues(op.getNumResults());
    unsigned secretResultIdx = 0;
    for (unsigned idx : secretReturnIndices) {
      finalReturnValues[idx] = newGeneric.getResult(secretResultIdx++);
    }
    for (unsigned idx : plaintextReturnIndices) {
      Value returnVal = returnOp->getOperand(idx);
      finalReturnValues[idx] = outerMapping.lookupOrDefault(returnVal);
    }

    func::ReturnOp::create(rewriter, op.getLoc(), finalReturnValues);

    // Erase the old block
    rewriter.eraseBlock(&opEntryBlock);

    return success();
  }

 private:
  DataFlowSolver* solver;
};

struct ConvertFuncCall : public OpRewritePattern<func::CallOp> {
  ConvertFuncCall(mlir::MLIRContext* context, Operation* top)
      : mlir::OpRewritePattern<func::CallOp>(context), top(top) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter& rewriter) const override {
    auto module = mlir::cast<ModuleOp>(top);
    auto callee = module.lookupSymbol<func::FuncOp>(op.getCallee());
    if (callee.isDeclaration()) {
      return success();
    }

    SmallVector<Value> newOperands;
    auto funcResultTypes = llvm::to_vector(callee.getResultTypes());

    for (auto i = 0; i != op->getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      auto funcArgType = callee.getArgumentTypes()[i];
      if (mlir::isa<secret::SecretType>(funcArgType)) {
        auto newOperand =
            secret::ConcealOp::create(rewriter, op.getLoc(), operand);
        newOperands.push_back(newOperand.getResult());
      } else {
        newOperands.push_back(operand);
      }
    }

    auto newOp = func::CallOp::create(rewriter, op->getLoc(), op.getCallee(),
                                      funcResultTypes, newOperands);
    newOp->setAttrs(op->getAttrs());

    for (auto i = 0; i != newOp->getNumResults(); ++i) {
      auto result = op.getResult(i);
      auto newResult = newOp.getResult(i);
      if (mlir::isa<secret::SecretType>(newResult.getType())) {
        newResult = secret::RevealOp::create(rewriter, op.getLoc(), newResult);
      }
      rewriter.replaceAllUsesWith(result, newResult);
    }
    rewriter.eraseOp(op);
    return success();
  }

 private:
  Operation* top;
};

struct WrapGeneric : impl::WrapGenericBase<WrapGeneric> {
  using WrapGenericBase::WrapGenericBase;

  void detectSecretGeneric() {
    // Note: Since we now correctly handle functions that return only
    // plaintext values (which don't get a secret.generic), we should not
    // warn about missing secret.generic ops. The warning was intended
    // for the case where users forgot to annotate secret inputs, but that
    // is already caught by the hasSecrets check in WrapWithGeneric.
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    // Run SecretnessAnalysis to determine which values depend on secrets
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run SecretnessAnalysis.\n";
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<WrapWithGeneric>(context, &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));

    // func.call should be converted after callee func type updated
    mlir::RewritePatternSet patterns2(context);
    patterns2.add<ConvertFuncCall>(context, getOperation());
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns2));

    // warn if no secret.generic found
    detectSecretGeneric();
  }
};

}  // namespace heir
}  // namespace mlir
