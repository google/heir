#ifndef LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
#define LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_

#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"        // from @llvm-project

namespace mlir {
namespace heir {

/// This class represents the multiplicative depth in an FHE program.
///
/// The multiplicative depth of a circuit refers to the maximum number of
/// multiplication gates on any path from a circuit input to an output. In a
/// more general FHE program, the multiplicative depth is the maximum number of
/// ciphertext-ciphertext or plaintext-ciphertext multiplication operations on
/// any use-def chain.
///
/// In FHE, multiplicative depth bounds the number of operations that can be
/// sequentially performed on a ciphertext before the noise introduced during
/// those operations renders the ciphertext undecryptable.
///
/// For example, the multiplicative depth of the following pseudo-code is 2:
///
///     %0 = arith.constant 0 : multiplicative depth = 0
///     %1 = FHE.mul %0 %0 : multiplicative depth = 1
///     %2 = FHE.add %0 %1 : multiplicative depth = 1
///     %3 = FHE.mul %2 %0 : multiplicative depth = 2
///
/// In more detail, the multiplicative depth for an arbitrary operation `op`
/// can be defined inductively as follows.
///
///     MulDepth(z) = case
///       mul_op(x, y):
///         max(MulDepth(x), MulDepth(y)) + 1
///       any_op(operands):
///         max(map(MulDepth, operands))

class MulDepth {
 public:
  MulDepth() : value(std::nullopt) {}
  explicit MulDepth(int64_t value) : value(value) {}
  ~MulDepth() = default;

  /// Whether the multiplicative depth is initialized. It can be uninitialized
  /// when the state hasn't been set during the analysis.
  bool isInitialized() const { return value.has_value(); }

  /// Get a known multiplicative depth.
  const int64_t &getValue() const {
    assert(isInitialized());
    return *value;
  }

  bool operator==(const MulDepth &rhs) const { return value == rhs.value; }

  /// Join two multiplicative depth.
  static MulDepth join(const MulDepth &lhs, const MulDepth &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    // If both are initialized, the larger value should be chosen for a sound
    // analysis of the multiplicative depth.
    return MulDepth{lhs.getValue() > rhs.getValue() ? lhs.getValue()
                                                    : rhs.getValue()};
  }

  void print(raw_ostream &os) const { os << "MulDepth(" << value << ")"; }

 private:
  /// The multiplicative depth, if known.
  std::optional<int64_t> value;

  friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diagnostic,
                                      const MulDepth &foo) {
    if (foo.isInitialized()) {
      return diagnostic << foo.getValue();
    }
    return diagnostic << "MulDepth(uninitialized)";
  }
};

inline raw_ostream &operator<<(raw_ostream &os, const MulDepth &v) {
  v.print(os);
  return os;
}

class MulDepthLattice : public dataflow::Lattice<MulDepth> {
 public:
  using Lattice::Lattice;
};

/// An analysis that identifies a multiplicative depth for an SSA value in a
/// program. This can be used by downstream passes to determine the
/// cryptographic parameters correctly.
///
/// Since the multiplicative depth can be calculated inductively, we implement
/// this analysis by using mlir's forward dataflow analysis.

class MulDepthAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<MulDepthLattice> {
 public:
  explicit MulDepthAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver) {}
  ~MulDepthAnalysis() override = default;
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // Given the computed results of the operation, update its operand lattice
  // values.
  void visitOperation(Operation *op, ArrayRef<const MulDepthLattice *> operands,
                      ArrayRef<MulDepthLattice *> results) override;

  // Instantiating the lattice to the uninitialized value
  void setToEntryState(MulDepthLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(MulDepth()));
  }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
