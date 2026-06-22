#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_CANDIDATE_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_CANDIDATE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Kernel/KernelName.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir::heir::rotom {

// The kind of HE kernel a candidate represents. Used only for debug labels and
// as a tie-breaker between otherwise equal-cost candidates.
enum class KernelKind {
  Tensor,
  BlockArgument,
  Yield,
  PassThrough,
  Elementwise,
  Generic,
  Transpose,
  Reduce,
  CollapseShape,
  ExpandShape,
  ExtractSlice,
  InsertSlice,
};

llvm::StringLiteral kernelKindName(KernelKind kind);

// The layout assignment of every value feeding a candidate: each such value
// mapped to (its chosen layout, that kernel's own local cost). A shared
// subexpression appears once, so summing the local costs never double-counts on
// a DAG. The assignment of the winning root candidate IS the output.
using Assignment = llvm::DenseMap<Value, std::pair<LayoutAttr, int64_t>>;

// A candidate layout for a single value: the value's layout, the dedup'd
// `accumulatedCost` (the sum over `assignment`), and the assignment itself.
// `localCost` is transient -- the value's own kernel cost, folded into
// `assignment`/`accumulatedCost` once the result value is known (see
// setCandidates). `localCost` and `assignment` are last so the positional
// aggregate-inits of the earlier fields keep working.
struct Candidate {
  LayoutAttr layout;
  int64_t accumulatedCost = 0;
  KernelKind kind = KernelKind::PassThrough;
  SmallVector<Value> operands;
  SmallVector<LayoutAttr> operandLayouts;
  std::optional<KernelName> kernel;
  int64_t localCost = 0;
  Assignment assignment;
};

// Sum of every kernel's local cost in the assignment (each value counted once).
int64_t accumulatedCostOf(const Assignment& assignment);
// Merges `from` into `into`. Returns false if they disagree on any value's
// layout (an inconsistent combination of sub-assignments on a DAG), leaving
// `into` partially merged -- the caller discards it.
bool mergeAssignments(Assignment& into, const Assignment& from);

// arith op classification.
bool isAddLike(Operation* op);
bool isAdd(Operation* op);
bool isMulLike(Operation* op);
std::optional<KernelName> selectRotomElementwiseKernel(Operation* op);

// Cost model (op weights live in CostModel.h / cost_model.json).
// `conversionCost` is the shift-network fallback used only when a layout cannot
// be lowered; `operationCost`/`genericOperationCost` score the compute from the
// aligned (compute) layout the operands are converted to -- one HE op per
// ciphertext.
int64_t conversionCost(ArrayRef<ConversionMove> moves);
int64_t operationCost(Operation* op, LayoutAttr alignedLayout);
int64_t genericOperationCost(linalg::GenericOp op, LayoutAttr alignedLayout);

// Ranking and deduplication. A candidate is "better" when it is cheaper, then
// when it carries a named kernel, then by a deterministic structural tie key.
std::string kernelKey(std::optional<KernelName> kernel);
std::string candidateTieKey(const Candidate& candidate);
bool isBetterCandidate(const Candidate& lhs, const Candidate& rhs);
SmallVector<Candidate> uniqueCandidates(ArrayRef<Candidate> candidates);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_CANDIDATE_H_
