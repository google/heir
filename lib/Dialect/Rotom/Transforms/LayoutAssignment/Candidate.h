#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_CANDIDATE_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_CANDIDATE_H_

#include <cstdint>
#include <optional>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Kernel/KernelName.h"
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

// A candidate layout assignment for a single value, together with the layouts
// its operands must take and the (optional) named kernel that implements it.
struct Candidate {
  LayoutAttr layout;
  int64_t cost = 0;
  KernelKind kind = KernelKind::PassThrough;
  SmallVector<Value> operands;
  SmallVector<LayoutAttr> operandLayouts;
  std::optional<KernelName> kernel;
};

// arith op classification.
bool isAddLike(Operation* op);
bool isAdd(Operation* op);
bool isMulLike(Operation* op);
std::optional<KernelName> selectRotomElementwiseKernel(Operation* op);

// Cost model. `layoutConversionCost` is a cheap proxy for realigning one layout
// onto another; `conversionCost` scores a set of slot-bit moves; `operationCost`
// and `genericOperationCost` score the compute itself.
int64_t layoutConversionCost(LayoutAttr from, LayoutAttr to);
int64_t conversionCost(ArrayRef<ConversionMove> moves);
int64_t operationCost(Operation* op, LayoutAttr layout);
int64_t genericOperationCost(linalg::GenericOp op, LayoutAttr layout);

// Ranking and deduplication. A candidate is "better" when it is cheaper, then
// when it carries a named kernel, then by a deterministic structural tie key.
std::string layoutKey(LayoutAttr layout);
std::string kernelKey(std::optional<KernelName> kernel);
std::string candidateTieKey(const Candidate& candidate);
bool isBetterCandidate(const Candidate& lhs, const Candidate& rhs);
SmallVector<Candidate> uniqueCandidates(ArrayRef<Candidate> candidates);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_CANDIDATE_H_
