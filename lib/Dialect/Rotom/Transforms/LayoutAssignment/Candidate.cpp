#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
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

bool isAddLike(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op);
}

bool isAdd(Operation* op) { return isa<arith::AddFOp, arith::AddIOp>(op); }

bool isMulLike(Operation* op) { return isa<arith::MulFOp, arith::MulIOp>(op); }

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
  key += candidate.hasRotomKernel ? "rotom" : "none";
  for (LayoutAttr operandLayout : candidate.operandLayouts) {
    key += ":";
    key += debugString(operandLayout);
  }
  return key;
}

bool isBetterCandidate(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.accumulatedCost != rhs.accumulatedCost)
    return lhs.accumulatedCost < rhs.accumulatedCost;
  if (lhs.hasRotomKernel != rhs.hasRotomKernel) {
    return lhs.hasRotomKernel;
  }
  return candidateTieKey(lhs) < candidateTieKey(rhs);
}

SmallVector<Candidate> uniqueCandidates(ArrayRef<Candidate> candidates) {
  SmallVector<Candidate> result;
  // Keyed on the dimension-merged canonical layout, so split forms of the
  // same packing ([0:2:8][0:8:1] vs [0:16:1]) compete as one entry and only
  // the best survives. The frontier itself is never truncated: every distinct
  // packing keeps its cheapest candidate.
  SmallVector<LayoutAttr> keys;
  for (const Candidate& candidate : candidates) {
    LayoutAttr key = mergeAdjacentLayoutDims(candidate.layout);
    size_t found = result.size();
    for (size_t i = 0; i < result.size(); ++i) {
      if (keys[i] == key &&
          result[i].hasRotomKernel == candidate.hasRotomKernel) {
        found = i;
        break;
      }
    }
    if (found == result.size()) {
      result.push_back(candidate);
      keys.push_back(key);
      continue;
    }
    if (isBetterCandidate(candidate, result[found])) {
      result[found] = candidate;
    }
  }
  llvm::sort(result, [](const Candidate& lhs, const Candidate& rhs) {
    return isBetterCandidate(lhs, rhs);
  });
  return result;
}

}  // namespace mlir::heir::rotom
