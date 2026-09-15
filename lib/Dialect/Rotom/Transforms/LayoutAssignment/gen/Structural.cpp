#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir::heir::rotom {

LogicalResult generateFunc(AssignmentContext& ctx, func::FuncOp op) {
  for (Value arg : op.getArguments()) ctx.seedValue(arg);
  return success();
}

LogicalResult generateSecretGeneric(AssignmentContext& ctx,
                                    secret::GenericOp op) {
  for (OpOperand& operand : op->getOpOperands()) {
    SmallVector<Candidate> operandCandidates =
        ctx.candidatesForValue(operand.get());
    if (operandCandidates.empty()) continue;
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    SmallVector<Candidate> blockArgCandidates;
    for (const Candidate& candidate : operandCandidates) {
      // The block argument is the operand routed into the region: same data,
      // same assignment, no extra cost.
      Candidate blockArgCandidate;
      blockArgCandidate.layout = candidate.layout;
      blockArgCandidate.kind = KernelKind::BlockArgument;
      blockArgCandidate.operands = {operand.get()};
      blockArgCandidate.operandLayouts = {candidate.layout};
      blockArgCandidate.assignment = candidate.assignment;
      blockArgCandidate.accumulatedCost =
          accumulatedCostOf(blockArgCandidate.assignment);
      blockArgCandidates.push_back(std::move(blockArgCandidate));
    }
    ctx.setCandidates(blockArg, blockArgCandidates);
  }
  return success();
}

LogicalResult generateYield(AssignmentContext& ctx, secret::YieldOp op) {
  auto generic = op->getParentOfType<secret::GenericOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    SmallVector<Candidate> yielded = ctx.candidatesForValue(operand.get());
    if (yielded.empty()) continue;
    SmallVector<Candidate> resultCandidates;
    for (const Candidate& candidate : yielded) {
      // The generic's result is the yielded value: same data, same assignment.
      Candidate resultCandidate;
      resultCandidate.layout = candidate.layout;
      resultCandidate.kind = KernelKind::Yield;
      resultCandidate.operands = {operand.get()};
      resultCandidate.operandLayouts = {candidate.layout};
      resultCandidate.assignment = candidate.assignment;
      resultCandidate.accumulatedCost =
          accumulatedCostOf(resultCandidate.assignment);
      resultCandidates.push_back(std::move(resultCandidate));
    }
    ctx.setCandidates(generic.getResult(operand.getOperandNumber()),
                      resultCandidates);
  }
  return success();
}

LogicalResult generatePassThrough(AssignmentContext& ctx, Operation* op) {
  SmallVector<Candidate> chosen =
      ctx.chooseCommonOperandCandidates(op, KernelKind::PassThrough);
  if (chosen.empty()) {
    for (Value result : op->getResults()) ctx.seedValue(result);
    return success();
  }
  ctx.assignResultsFromCandidates(op, chosen);
  return success();
}

}  // namespace mlir::heir::rotom
