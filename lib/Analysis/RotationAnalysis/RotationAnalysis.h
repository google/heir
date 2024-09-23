#ifndef LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_
#define LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_

#include <set>

#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {
namespace rotation_analysis {

// A PartialReduction represents a subset of an arithmetic op tree that reduces
// values within a tensor to a scalar (present in index zero of the result
// tensor).
//
// It is "partial" in the sense that it may not reduce across all elements of a
// tensor, and it is used in the analysis to accumulate reduced tensor indices
// across the IR.
//
// It also stores a reference to the SSA value that identifes the "end" of the
// computation (i.e., the SSA value that contains the result of the reduction).
class PartialReduction {
 public:
  bool empty() const { return accessedIndices.empty(); }

  void addRotation(int64_t index) { accessedIndices.insert(index); }

  // Returns true if the accessed indices constitute all indices of the reduced
  // tensor.
  bool isComplete() const {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
    assert(tensorType &&
           "Internal state of RotationAnalysis is broken; tensor must have a "
           "ranked tensor type");

    // std::set is ordered, so min/max is first/last element of the set
    int64_t minIndex = *accessedIndices.begin();
    int64_t maxIndex = *accessedIndices.rbegin();
    return minIndex == 0 && maxIndex == tensorType.getShape()[0] - 1 &&
           accessedIndices.size() == (size_t)tensorType.getShape()[0];
  }

  const std::set<int64_t> &getAccessedIndices() const {
    return accessedIndices;
  }

  Value getTensor() const { return tensor; }

  Value getRoot() const { return root; }

  void print(raw_ostream &os) const {
    os << "{ opName: " << (opName.has_value() ? opName->getStringRef() : "None")
       << "; " << " tensor: " << tensor << "; " << "rotations: [";
    for (auto index : accessedIndices) {
      os << index << ", ";
    }
    os << "]; root: " << root << "; }";
  }

  // Construct a "leaf" of a reduction, i.e., a PartialReduction that represents
  // no operations applied to a starting tensor SSA value.
  static PartialReduction initializeFromValue(Value tensor) {
    PartialReduction reduction;
    reduction.tensor = tensor;
    reduction.root = tensor;
    reduction.opName = std::nullopt;
    // In the FHE world, the only extractible element (without a rotation) of a
    // packed ciphertext is the constant term, i.e., the first element of the
    // tensor. So a tensor by itself is always considered a reduction by that
    // first element.
    reduction.addRotation(0);

    LLVM_DEBUG(llvm::dbgs()
               << "Initializing at " << tensor << " with rotations [0]\n");
    return reduction;
  }

  // Shift the rotation indices by the given amount. This helps in a situation
  // where an IR repeatedly rotates by 1, to ensure that rotations accumulate
  // like {1, 2, 3, ...} rather than {1, 1, 1, ...}
  static PartialReduction rotate(const PartialReduction &lhs,
                                 const int64_t shift, Value result) {
    LLVM_DEBUG({
      llvm::dbgs() << "Rotating\n\t";
      lhs.print(llvm::dbgs());
      llvm::dbgs() << " by " << shift;
    });
    PartialReduction shifted;
    shifted.tensor = lhs.tensor;
    shifted.opName = lhs.opName;
    shifted.root = result;
    int64_t size =
        llvm::cast<RankedTensorType>(lhs.tensor.getType()).getShape()[0];
    assert(!lhs.accessedIndices.empty() &&
           "Internal state of RotationAnalysis is broken; empty rotation sets "
           "should be impossible");
    for (auto index : lhs.accessedIndices) {
      shifted.addRotation((index + shift) % size);
    }
    LLVM_DEBUG({
      llvm::dbgs() << " to\n\t";
      shifted.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    return shifted;
  }

  // Determine if two PartialRotations are legal to join at an op whose
  // OperationName is given.
  static bool canJoin(const PartialReduction &lhs, const PartialReduction &rhs,
                      OperationName opName) {
    if (lhs.tensor != rhs.tensor) {
      return false;
    }

    // If neither of the lhs and rhs ops are set, then any op is legal.
    if (lhs.opName.has_value() || rhs.opName.has_value()) {
      // Otherwise, if both ops are set, then they must agree with each other
      // and the new op.
      if (lhs.opName.has_value() && rhs.opName.has_value() &&
          (*lhs.opName != *rhs.opName || *lhs.opName != opName)) {
        return false;
      }

      // Otherwise, at least one of lhs and rhs must have a set op name, and it
      // must agree with the new op.
      auto materializedOpName =
          lhs.opName.has_value() ? *lhs.opName : *rhs.opName;
      if (materializedOpName != opName) {
        return false;
      }
    }

    // If the two partial reductions have access indices in common, then they
    // cannot be joined because some indices would be contributing multiple
    // times to the overall reduction. Maybe we could improve this in the
    // future so that we could handle a kind of reduction that sums the same
    // index twice, but likely it is better to account for that in a different
    // fashion.
    auto smaller =
        lhs.accessedIndices.size() < rhs.accessedIndices.size() ? lhs : rhs;
    auto larger =
        lhs.accessedIndices.size() >= rhs.accessedIndices.size() ? lhs : rhs;
    return std::all_of(smaller.accessedIndices.begin(),
                       smaller.accessedIndices.end(), [&](int64_t index) {
                         return larger.accessedIndices.count(index) == 0;
                       });
  }

  // Join two partial reductions. This assumes the lhs and rhs have already
  // been checked to have compatible tensors and opNames via canJoin.
  static PartialReduction join(const PartialReduction &lhs,
                               const PartialReduction &rhs, Value newRoot,
                               OperationName opName) {
    assert(!lhs.accessedIndices.empty() &&
           "Internal state of RotationAnalysis is broken; empty rotation sets "
           "should be impossible");
    assert(!rhs.accessedIndices.empty() &&
           "Internal state of RotationAnalysis is broken; empty rotation sets "
           "should be impossible");

    PartialReduction merged;
    merged.tensor = lhs.tensor;
    merged.root = newRoot;
    merged.opName = opName;
    for (auto index : lhs.accessedIndices) {
      merged.addRotation(index);
    }
    for (auto index : rhs.accessedIndices) {
      merged.addRotation(index);
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Joining\n\t";
      lhs.print(llvm::dbgs());
      llvm::dbgs() << " and\n\t";
      rhs.print(llvm::dbgs());
      llvm::dbgs() << " to get\n\t";
      merged.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    return merged;
  }

 private:
  // The SSA value being reduced
  Value tensor;

  // The root of the reduction tree constructed so far, e.g., the result of the
  // last op in a linear chain of reduction operations. During
  // rotate-and-reduce, this represents the final SSA value that is replaced by
  // an optimized set of rotations.
  Value root;

  // The operation performed in the reduction.
  //
  // Set to std::nullopt if no binary operation is applied (i.e., the reduction
  // is a raw tensor at the leaf of a reduction tree).
  std::optional<OperationName> opName;

  // The set of indices of `tensor` accumulated by the reduction so far.
  //
  // There is likely a data structure that can more efficiently represent a set
  // of intervals of integers, which properly merges adjacent intervals as
  // values are added. Java/Guava has RangeSet, and boost has interval_set.
  // For now we use std::set which is implemented as a binary tree and ordered
  // by the index values.
  std::set<int64_t> accessedIndices;
};

inline raw_ostream &operator<<(raw_ostream &os, const PartialReduction &v) {
  v.print(os);
  return os;
}

/// An analysis that identifies, for each tensor-typed SSA value, the set of
/// partial reductions of associative, commutative binary arithmetic operations
/// that reduce it to a scalar via tensor_ext.rotate ops.
class RotationAnalysis {
 public:
  // The constructor requires a DataFlowSolver initialized with a sparse
  // constant propagation analysis, which is used to determine the static
  // values of rotation shifts.
  RotationAnalysis(const DataFlowSolver &solver) : solver(solver) {}
  ~RotationAnalysis() = default;

  void run(Operation *op);

  /// Add partial reduction
  void addPartialReduction(PartialReduction reduction) {
    rootToPartialReductions[reduction.getRoot()].emplace_back(reduction);
  }

  /// Add a tensor value as the start of a new reduction to the internal
  /// reduction mappings.
  void initializeFromValueIfTensor(Value value) {
    if (RankedTensorType tensorType =
            mlir::dyn_cast<RankedTensorType>(value.getType())) {
      addPartialReduction(PartialReduction::initializeFromValue(value));
    }
  }

  const std::vector<PartialReduction> &getRootedReductionsAt(
      Value value) const {
    return rootToPartialReductions.at(value);
  }

  bool containsRootedReductions(Value value) const {
    return rootToPartialReductions.contains(value);
  }

 private:
  // The constant propagation analysis used to determine the static values of
  // rotation shifts.
  const DataFlowSolver &solver;

  // A mapping from a root of a PartialReduction to its PartitalReduction. Note
  // each tensor SSA value can be the root of many partial reductions.
  llvm::DenseMap<Value, std::vector<PartialReduction>> rootToPartialReductions;
};

}  // namespace rotation_analysis
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_
