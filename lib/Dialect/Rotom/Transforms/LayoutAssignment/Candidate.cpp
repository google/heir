#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Kernel/KernelName.h"
#include "llvm/include/llvm/ADT/STLExtras.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/include/mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

namespace mlir::heir::rotom {

int64_t accumulatedCostOf(const Assignment& assignment) {
  int64_t total = 0;
  for (const auto& entry : assignment) total += entry.second.second;
  return total;
}

bool mergeAssignments(Assignment& into, const Assignment& from) {
  for (const auto& entry : from) {
    auto it = into.find(entry.first);
    if (it == into.end()) {
      into.insert(entry);
    } else if (it->second.first != entry.second.first) {
      return false;
    }
  }
  return true;
}

llvm::StringLiteral kernelKindName(KernelKind kind) {
  switch (kind) {
    case KernelKind::Tensor:
      return "tensor";
    case KernelKind::BlockArgument:
      return "block_arg";
    case KernelKind::Yield:
      return "yield";
    case KernelKind::PassThrough:
      return "pass_through";
    case KernelKind::Elementwise:
      return "elementwise";
    case KernelKind::Generic:
      return "generic";
    case KernelKind::Transpose:
      return "transpose";
    case KernelKind::Reduce:
      return "reduce";
    case KernelKind::Matmul:
      return "matmul";
    case KernelKind::CollapseShape:
      return "collapse_shape";
    case KernelKind::ExpandShape:
      return "expand_shape";
    case KernelKind::ExtractSlice:
      return "extract_slice";
    case KernelKind::InsertSlice:
      return "insert_slice";
  }
  llvm_unreachable("unknown kernel kind");
}

std::string kernelKey(std::optional<KernelName> kernel) {
  return kernel ? debugString(*kernel) : "none";
}

bool isAddLike(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op);
}

bool isAdd(Operation* op) { return isa<arith::AddFOp, arith::AddIOp>(op); }

bool isMulLike(Operation* op) { return isa<arith::MulFOp, arith::MulIOp>(op); }

std::optional<KernelName> selectRotomElementwiseKernel(Operation* op) {
  // Subtraction is additive (depth-0, both operands at the shared layout), so
  // it shares the RotomAdd kernel; only the emitted arith op differs.
  if (isAddLike(op)) return KernelName::RotomAdd;
  if (isMulLike(op)) return KernelName::RotomMul;
  return std::nullopt;
}

// Cost of running `op` once its operands are aligned to `alignedLayout` (the
// compute layout): one HE op per ciphertext, so weight x numCt(alignedLayout).
int64_t operationCost(Operation* op, LayoutAttr alignedLayout) {
  int64_t numCt = layoutNumCiphertexts(alignedLayout);
  const RotomCostModel& model = getCostModel();
  if (isAddLike(op)) return model.add * numCt;
  if (isMulLike(op)) return model.ciphertextMultiply * numCt;
  return 0;
}

// A linalg.generic's cost is its body's per-ciphertext op costs summed at the
// aligned (compute) layout its inputs are converted to.
int64_t genericOperationCost(linalg::GenericOp op, LayoutAttr alignedLayout) {
  int64_t cost = 0;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    cost += operationCost(&innerOp, alignedLayout);
  }
  return cost;
}

std::string candidateTieKey(const Candidate& candidate) {
  std::string key = kernelKindName(candidate.kind).str();
  key += ":";
  key += debugString(candidate.layout);
  key += ":kernel=";
  key += kernelKey(candidate.kernel);
  for (LayoutAttr operandLayout : candidate.operandLayouts) {
    key += ":";
    key += debugString(operandLayout);
  }
  return key;
}

bool isBetterCandidate(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.accumulatedCost != rhs.accumulatedCost)
    return lhs.accumulatedCost < rhs.accumulatedCost;
  if (lhs.kernel.has_value() != rhs.kernel.has_value()) {
    return lhs.kernel.has_value();
  }
  return candidateTieKey(lhs) < candidateTieKey(rhs);
}

SmallVector<Candidate> uniqueCandidates(ArrayRef<Candidate> candidates) {
  SmallVector<Candidate> result;
  for (const Candidate& candidate : candidates) {
    auto it = llvm::find_if(result, [&](const Candidate& existing) {
      return existing.layout == candidate.layout &&
             existing.kernel == candidate.kernel;
    });
    if (it == result.end()) {
      result.push_back(candidate);
      continue;
    }
    if (isBetterCandidate(candidate, *it)) {
      *it = candidate;
    }
  }
  llvm::sort(result, [](const Candidate& lhs, const Candidate& rhs) {
    return isBetterCandidate(lhs, rhs);
  });
  return result;
}

}  // namespace mlir::heir::rotom
