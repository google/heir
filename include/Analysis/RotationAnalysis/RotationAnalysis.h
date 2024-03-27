#ifndef INCLUDE_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_
#define INCLUDE_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_

#include <unordered_set>

#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {
namespace rotation_analysis {

// A wrapper around a mapping from a single tensor SSA value to a set of its
// access indices.
class RotationSets {
 public:
  enum class Status {
    // The tensor value has not been set
    Uninitialized,

    // The rotation set is in a normal state.
    Normal,

    // The rotation set has a property that makes it invalid for later
    // optimizations:
    //
    //  - It involves operations touch more than one source tensor (not
    //    including value-semantic outputs)
    Overdetermined

  };

 public:
  RotationSets() = default;
  ~RotationSets() = default;

  // Clear the member data, i.e., set the value back to an uninitialized
  // state.
  void clear() {
    accessedIndices.clear();
    status = Status::Uninitialized;
  }

  bool empty() const { return accessedIndices.empty(); }

  bool isOverdetermined() const { return status == Status::Overdetermined; }

  bool isUninitialized() const { return status == Status::Uninitialized; }

  void addRotation(int64_t index) { accessedIndices.insert(index); }

  bool operator==(const RotationSets &rhs) const {
    return tensor == rhs.tensor && status == rhs.status &&
           accessedIndices == rhs.accessedIndices;
  }

  const std::unordered_set<int64_t> &getAccessedIndices() const {
    return accessedIndices;
  }

  Value getTensor() const { return tensor; }

  void print(raw_ostream &os) const {
    os << tensor << ": [";
    for (auto index : accessedIndices) {
      os << index << ", ";
    }
    os << "]";
  }

  static RotationSets overdetermined() {
    RotationSets sets;
    sets.status = Status::Overdetermined;
    return sets;
  }

  static RotationSets from(Value tensor) {
    RotationSets sets;
    if (!tensor.getType().isa<RankedTensorType>()) {
      sets.status = Status::Uninitialized;
      return sets;
    }

    sets.status = Status::Normal;
    sets.tensor = tensor;
    if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
      sets.addRotation(0);
    }
    return sets;
  }

  // Shift the rotation indices by the given amount. This helps in a situation
  // where an IR repeatedly rotates by 1, to ensure that rotations accumulate
  // like {1, 2, 3, ...} rather than {1, 1, 1, ...}
  static RotationSets rotate(const RotationSets &lhs, const int64_t shift) {
    if (lhs.status == Status::Overdetermined) {
      return overdetermined();
    }

    RotationSets shifted;
    shifted.status = Status::Normal;
    shifted.tensor = lhs.tensor;
    int64_t size =
        llvm::cast<RankedTensorType>(lhs.tensor.getType()).getShape()[0];
    for (auto index : lhs.accessedIndices) {
      shifted.addRotation((index + shift) % size);
    }
    return shifted;
  }

  static RotationSets join(const RotationSets &lhs, const RotationSets &rhs) {
    if (lhs.status == Status::Overdetermined ||
        rhs.status == Status::Overdetermined) {
      return overdetermined();
    }

    if (rhs.status == Status::Uninitialized || rhs.accessedIndices.empty())
      return lhs;
    if (lhs.status == Status::Uninitialized || lhs.accessedIndices.empty())
      return rhs;

    if (lhs.tensor != rhs.tensor) {
      LLVM_DEBUG({
        llvm::dbgs() << "Joining rotations of different tensors: " << lhs.tensor
                     << " and " << rhs.tensor << "\n";
      });
      return overdetermined();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Joining :" << lhs.tensor << " and " << rhs.tensor
                   << "\n";
    });
    RotationSets merged;
    merged.status = Status::Normal;
    merged.tensor = lhs.tensor;
    for (auto index : lhs.accessedIndices) {
      merged.addRotation(index);
    }
    for (auto index : rhs.accessedIndices) {
      merged.addRotation(index);
    }
    return merged;
  }

  // Assuming two not-overdetermined rotation sets, compute the overlap in
  // their access indices.
  static RotationSets overlap(const RotationSets &lhs,
                              const RotationSets &rhs) {
    assert(!lhs.isOverdetermined() && !rhs.isOverdetermined() &&
           "Expected inputs to RotationSets::overlap to be not overdetermined");
    if (lhs.status == Status::Uninitialized || lhs.empty()) {
      return lhs;
    }

    if (rhs.status == Status::Uninitialized || rhs.empty()) {
      return rhs;
    }

    RotationSets merged;
    merged.status = Status::Normal;
    merged.tensor = lhs.tensor;
    for (auto index : lhs.accessedIndices) {
      if (rhs.accessedIndices.count(index)) merged.addRotation(index);
    }
    return merged;
  }

 private:
  /// The accessed indices of a single SSA value of tensor type.
  Value tensor;

  // There is likely a data structure that can more efficiently represent a set
  // of intervals of integers, which properly merges adjacent intervals as
  // values are added. Java/Guava has RangeSet, and boost has interval_set.
  std::unordered_set<int64_t> accessedIndices;
  Status status = Status::Uninitialized;
};

inline raw_ostream &operator<<(raw_ostream &os, const RotationSets &v) {
  v.print(os);
  return os;
}

class RotationLattice : public dataflow::Lattice<RotationSets> {
 public:
  using Lattice::Lattice;
};

/// An analysis that identifies, for each SSA value, the set of underlying
/// tensors and rotations of those tensors, provided constant rotation shifts
/// can be determined.
class RotationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<RotationLattice> {
 public:
  explicit RotationAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver) {}
  ~RotationAnalysis() override = default;
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // Given the computed results of the operation, update its operand lattice
  // values.
  void visitOperation(Operation *op, ArrayRef<const RotationLattice *> operands,
                      ArrayRef<RotationLattice *> results) override;

  void setToEntryState(RotationLattice *lattice) override;
};

}  // namespace rotation_analysis
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_
