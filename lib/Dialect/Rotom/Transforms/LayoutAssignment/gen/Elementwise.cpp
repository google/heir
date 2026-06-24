#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/ValueUtils.h"
#include "lib/Kernel/KernelName.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::rotom {

// A linalg.generic is elementwise when every operand is read with the identity
// indexing map under purely parallel iteration -- no broadcast, reduction, or
// permutation -- so a layout can pass through it unchanged.
static bool isElementwiseGeneric(linalg::GenericOp op) {
  for (AffineMap map : op.getIndexingMapsArray()) {
    if (!map.isIdentity()) return false;
  }
  for (utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) return false;
  }
  return true;
}

// True when the body is a single add-like op (ignoring the yield and any
// constants), i.e. the generic computes an elementwise addition.
static bool hasAddLikeBody(linalg::GenericOp op) {
  bool foundAddLikeOp = false;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    if (!isAddLike(&innerOp)) return false;
    foundAddLikeOp = true;
  }
  return foundAddLikeOp;
}

LogicalResult generateElementwise(AssignmentContext& ctx, Operation* op) {
  if (op->getNumOperands() == 2) {
    std::optional<KernelName> rotomKernel = selectRotomElementwiseKernel(op);
    SmallVector<Value> operands = {op->getOperand(0), op->getOperand(1)};
    SmallVector<Candidate> kernels = ctx.chooseElementwiseKernels(
        operands, KernelKind::Elementwise,
        [&](LayoutAttr layout) { return operationCost(op, layout); },
        rotomKernel);
    if (!kernels.empty()) {
      ctx.assignResultsFromCandidates(op, kernels);
      return success();
    }
  }

  SmallVector<Candidate> chosen =
      ctx.chooseCommonOperandCandidates(op, KernelKind::Elementwise);
  ctx.assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult generateLinalgGeneric(AssignmentContext& ctx,
                                    linalg::GenericOp op) {
  if (!isElementwiseGeneric(op)) return generatePassThrough(ctx, op);
  if (hasAddLikeBody(op) && op.getInputs().size() == 2) {
    SmallVector<Value> operands = {op.getInputs()[0], op.getInputs()[1]};
    SmallVector<Candidate> kernels = ctx.chooseElementwiseKernels(
        operands, KernelKind::Generic,
        [&](LayoutAttr layout) { return genericOperationCost(op, layout); });
    if (!kernels.empty()) {
      ctx.assignResultsFromCandidates(op, kernels);
      return success();
    }
  }

  SmallVector<Value> operands;
  SmallVector<SmallVector<Candidate>> candidateSets;
  for (Value operand : op->getOperands()) {
    if (!isTensorLike(operand)) continue;
    SmallVector<Candidate> operandCandidates = ctx.candidatesForValue(operand);
    if (operandCandidates.empty()) continue;
    operands.push_back(operand);
    candidateSets.push_back(operandCandidates);
  }
  SmallVector<Candidate> chosen = chooseCommonCandidates(
      operands, candidateSets, KernelKind::Generic,
      [&](LayoutAttr layout) { return genericOperationCost(op, layout); },
      [&ctx](LayoutAttr from, LayoutAttr to) {
        return ctx.cachedConversionCost(from, to);
      });
  ctx.assignResultsFromCandidates(op, chosen);
  return success();
}

}  // namespace mlir::heir::rotom
