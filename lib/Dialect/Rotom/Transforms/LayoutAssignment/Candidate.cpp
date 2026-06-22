#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Kernel/KernelName.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir::heir::rotom {

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

std::string layoutKey(LayoutAttr layout) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << layout;
  return storage;
}

std::string kernelKey(std::optional<KernelName> kernel) {
  if (!kernel) return "none";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << *kernel;
  return storage;
}

bool isAddLike(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op);
}

bool isAdd(Operation* op) { return isa<arith::AddFOp, arith::AddIOp>(op); }

bool isMulLike(Operation* op) { return isa<arith::MulFOp, arith::MulIOp>(op); }

std::optional<KernelName> selectRotomElementwiseKernel(Operation* op) {
  if (isAdd(op)) return KernelName::RotomAdd;
  if (isMulLike(op)) return KernelName::RotomMul;
  return std::nullopt;
}

int64_t layoutConversionCost(LayoutAttr from, LayoutAttr to) {
  if (from == to) return 0;
  return 4 + std::abs(layoutNumCiphertexts(from) - layoutNumCiphertexts(to));
}

// Proxy cost of converting one operand's layout onto another, measured by the
// slot bits that must move. Zero when the layouts are already aligned (no
// moves). A later stage replaces this with the Vos-Vos-Erkin shift-network cost
// once the moves are lowered through tensor_ext.convert_layout.
int64_t conversionCost(ArrayRef<ConversionMove> moves) {
  if (moves.empty()) return 0;
  return 4 + static_cast<int64_t>(moves.size());
}

int64_t operationCost(Operation* op, LayoutAttr layout) {
  int64_t numCt = layoutNumCiphertexts(layout);
  if (isAddLike(op)) return numCt;
  if (isMulLike(op)) return 10 * numCt;
  return 0;
}

int64_t genericOperationCost(linalg::GenericOp op, LayoutAttr layout) {
  int64_t cost = 0;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    cost += operationCost(&innerOp, layout);
  }
  return cost;
}

std::string candidateTieKey(const Candidate& candidate) {
  std::string key = kernelKindName(candidate.kind).str();
  key += ":";
  key += layoutKey(candidate.layout);
  key += ":kernel=";
  key += kernelKey(candidate.kernel);
  for (LayoutAttr operandLayout : candidate.operandLayouts) {
    key += ":";
    key += layoutKey(operandLayout);
  }
  return key;
}

bool isBetterCandidate(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
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
