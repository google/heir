#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_ASSIGNMENTCONTEXT_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_ASSIGNMENTCONTEXT_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Kernel/KernelName.h"
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::heir::rotom {

// The narrow interface the per-op candidate generators use to read and write
// the candidate map and query costs, so each generator depends only on this
// rather than on the LayoutAssignment pass struct. The pass implements it.
class AssignmentContext {
 public:
  virtual ~AssignmentContext() = default;

  // Seed `value` from its rotom.seed attribute (no-op if already processed).
  virtual void seedValue(Value value) = 0;

  // The candidate layouts assigned to `value` (seeding it first if needed).
  virtual SmallVector<Candidate> candidatesForValue(Value value) = 0;

  // Record `newCandidates` as the candidates for `value`, after validating and
  // folding each into its assignment.
  virtual void setCandidates(Value value,
                             ArrayRef<Candidate> newCandidates) = 0;

  // Set the chosen candidate set on every tensor result of `op`.
  virtual void assignResultsFromCandidates(Operation* op,
                                           ArrayRef<Candidate> chosen) = 0;

  // Bring all of `op`'s operands onto a common result layout (the generic
  // N-operand combiner).
  virtual SmallVector<Candidate> chooseCommonOperandCandidates(
      Operation* op, KernelKind kind) = 0;

  // The specialized binary convert-then-compute combiner (add/sub/mul).
  virtual SmallVector<Candidate> chooseElementwiseKernels(
      ArrayRef<Value> operands, KernelKind kind,
      function_ref<int64_t(LayoutAttr)> computeCostFn,
      std::optional<KernelName> rotomKernel = std::nullopt) = 0;

  // Real (cached) rotation cost of converting `from` onto `to`.
  virtual int64_t cachedConversionCost(LayoutAttr from, LayoutAttr to) = 0;
};

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_ASSIGNMENTCONTEXT_H_
