#ifndef LIB_ANALYSIS_TARGETSLOTANALYSIS_TARGETSLOTANALYSIS_H_
#define LIB_ANALYSIS_TARGETSLOTANALYSIS_TARGETSLOTANALYSIS_H_

#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace target_slot_analysis {

/// A target slot is an identification of a downstream tensor index at which an
/// SSA value will be used. To make the previous sentence even mildly
/// comprehensible, consider it in the following example.
///
///     %c3 = arith.constant 3 : index
///     %c4 = arith.constant 4 : index
///     %c11 = arith.constant 11 : index
///     %c15 = arith.constant 15 : index
///     %v11 = tensor.extract %arg1[%c11] : tensor<16xi32>
///     %v15 = tensor.extract %arg1[%c15] : tensor<16xi32>
///     %1 = arith.addi %v11, %v15: i32
///     %v3 = tensor.extract %arg1[%c3] : tensor<16xi32>
///     %2 = arith.addi %v3, %1 : i32
///     %inserted = tensor.insert %2 into %output[%c4] : tensor<16xi32>
///
/// In vectorized FHE schemes like BGV, the computation model does not
/// efficiently support extracting values at particular indices; instead, it
/// supports SIMD additions of entire vectors, and cyclic rotations of vectors
/// by constant shifts. To optimize the above computation, we want to convert
/// the extractions to rotations, and minimize rotations as much as possible.
///
/// A naive conversion convert tensor.extract %arg1[Z] to arith.rotate %arg1,
/// Z, always placing the needed values in the zero-th slot. However, the last
/// line above indicates that the downstream dependencies of these computations
/// are ultimately needed in slot 4 of the %output tensor. So one could reduce
/// the number of rotations by rotating instead to slot 4, so that the final
/// rotation is not needed.
///
/// This analysis identifies that downstream insertion index, and propagates it
/// backward through the IR to attach it to each SSA value, enabling later
/// optimization passes to access it easily.
///
/// As it turns out, if the IR is well-structured, such as an unrolled affine
/// for loop with simple iteration strides, then aligning to target slots in
/// this way leads to many common sub-expressions that can be eliminated. Cf.
/// the insert-rotate pass for more on that.

class TargetSlot {
 public:
  TargetSlot() : value(std::nullopt) {}
  TargetSlot(int64_t value) : value(value) {}
  ~TargetSlot() = default;

  /// Whether the slot target is initialized. It can be uninitialized when the
  /// state hasn't been set during the analysis.
  bool isInitialized() const { return value.has_value(); }

  /// Get a known slot target.
  const int64_t &getValue() const {
    assert(isInitialized());
    return *value;
  }

  bool operator==(const TargetSlot &rhs) const { return value == rhs.value; }

  /// Join two target slots.
  static TargetSlot join(const TargetSlot &lhs, const TargetSlot &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    // If they are both initialized, use an arbitrary deterministic rule to
    // select one. A more sophisticated analysis could try to determine which
    // slot is more likely to lead to beneficial optimizations.
    return TargetSlot{lhs.getValue() < rhs.getValue() ? lhs.getValue()
                                                      : rhs.getValue()};
  }

  void print(raw_ostream &os) const { os << value; }

 private:
  /// The target slot, if known.
  std::optional<int64_t> value;

  friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diagnostic,
                                      const TargetSlot &foo) {
    if (foo.isInitialized()) {
      return diagnostic << foo.getValue();
    }
    return diagnostic << "uninitialized";
  }
};

inline raw_ostream &operator<<(raw_ostream &os, const TargetSlot &v) {
  v.print(os);
  return os;
}

class TargetSlotLattice : public dataflow::Lattice<TargetSlot> {
 public:
  using Lattice::Lattice;
};

/// An analysis that identifies a target slot for an SSA value in a program.
/// This is used by downstream passes to determine how to align rotations in
/// vectorized FHE schemes.
///
/// We use a backward dataflow analysis because the target slot propagates
/// backward from its final use to the arithmetic operations at which rotations
/// can be optimized.
class TargetSlotAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<TargetSlotLattice> {
 public:
  explicit TargetSlotAnalysis(
      DataFlowSolver &solver, SymbolTableCollection &symbolTable,
      // The dataflow solver is a private member of the base analysis
      // class, so if we want to access it we have to get it explicitly from
      // the caller. It's required that this solver is pre-loaded with a
      // SparseConstantPropagation analysis. I'd like a better way to do
      // this: maybe pass a callback?
      const DataFlowSolver *sccpAnalysis)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable),
        sccpAnalysis(sccpAnalysis) {}
  ~TargetSlotAnalysis() override = default;
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  // Given the computed results of the operation, update its operand lattice
  // values.
  LogicalResult visitOperation(
      Operation *op, ArrayRef<TargetSlotLattice *> operands,
      ArrayRef<const TargetSlotLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override {};
  void visitCallOperand(OpOperand &operand) override {};
  void setToExitState(TargetSlotLattice *lattice) override {};

 private:
  const DataFlowSolver *sccpAnalysis;
};

}  // namespace target_slot_analysis
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_TARGETSLOTANALYSIS_TARGETSLOTANALYSIS_H_
