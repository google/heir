#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

LogicalResult generateTranspose(AssignmentContext& ctx,
                                linalg::TransposeOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType) return generatePassThrough(ctx, op);

  SmallVector<int64_t> oldToNew(inputType.getRank(), -2);
  for (auto [outputDim, inputDim] : llvm::enumerate(op.getPermutation())) {
    if (inputDim < 0 || inputDim >= inputType.getRank()) {
      return generatePassThrough(ctx, op);
    }
    oldToNew[inputDim] = static_cast<int64_t>(outputDim);
  }

  SmallVector<Candidate> inputCandidates =
      ctx.candidatesForValue(op.getInput());
  SmallVector<Candidate> transposed = remapCandidates(
      op.getInput(), inputCandidates, oldToNew, KernelKind::Transpose);
  ctx.assignResultsFromCandidates(op, transposed);
  return success();
}

LogicalResult generateReduction(AssignmentContext& ctx, linalg::ReduceOp op) {
  for (auto [input, result] : llvm::zip(op.getInputs(), op.getResults())) {
    SmallVector<Candidate> inputCandidates = ctx.candidatesForValue(input);
    if (inputCandidates.empty()) continue;

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) continue;

    std::optional<SmallVector<int64_t>> oldToNew =
        getReductionDimMap(inputType.getRank(), op.getDimensions());
    if (!oldToNew) continue;

    SmallVector<Candidate> reduced =
        remapCandidates(input, inputCandidates, *oldToNew, KernelKind::Reduce);
    for (Candidate& candidate : reduced) {
      // A reduction sums its input ciphertexts, so its local cost is set by the
      // aligned INPUT layout (operandLayouts[0]) -- one add per input
      // ciphertext -- not the smaller reduced output layout.
      if (candidate.operandLayouts.empty()) continue;
      int64_t inputNumCt = layoutNumCiphertexts(candidate.operandLayouts[0]);
      candidate.localCost += inputNumCt * getCostModel().add;
    }
    ctx.setCandidates(result, reduced);
  }
  return success();
}

}  // namespace mlir::heir::rotom
