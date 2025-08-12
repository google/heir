#ifndef LIB_ANALYSIS_DIMENSIONANALYSIS_DIMENSIONANALYSIS_H_
#define LIB_ANALYSIS_DIMENSIONANALYSIS_DIMENSIONANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// This analysis should be used after --mlir-to-secret-arithmetic
// but before --secret-distribute-generic
// where a whole secret::GenericOp is assumed

class DimensionState {
 public:
  using DimensionType = int;

  DimensionState() : dimension(std::nullopt) {}
  explicit DimensionState(DimensionType dimension) : dimension(dimension) {}
  ~DimensionState() = default;

  DimensionType getDimension() const {
    assert(isInitialized());
    return dimension.value();
  }
  DimensionType get() const { return getDimension(); }

  bool operator==(const DimensionState& rhs) const {
    return dimension == rhs.dimension;
  }

  bool isInitialized() const { return dimension.has_value(); }

  static DimensionState join(const DimensionState& lhs,
                             const DimensionState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return DimensionState{std::max(lhs.getDimension(), rhs.getDimension())};
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "DimensionState(" << dimension.value() << ")";
    } else {
      os << "DimensionState(uninitialized)";
    }
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const DimensionState& state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<DimensionType> dimension;
};

class DimensionLattice : public dataflow::Lattice<DimensionState> {
 public:
  using Lattice::Lattice;
};

class DimensionAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<DimensionLattice>,
      public SecretnessAnalysisDependent<DimensionAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<DimensionAnalysis>;

  void setToEntryState(DimensionLattice* lattice) override {
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice, lattice->join(DimensionState(2)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(DimensionState()));
  }

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const DimensionLattice*> operands,
                               ArrayRef<DimensionLattice*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const DimensionLattice*> argumentLattices,
                         ArrayRef<DimensionLattice*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

/// Back propagate dimension to plaintext operands.
///
/// This analysis should be run after the (forward) DimensionAnalysis
/// where the dimension of all the secret Values are determined.
/// Then, this analysis will find ct-pt pair and determine the
/// dimension of the pt Value.
class DimensionAnalysisBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<DimensionLattice>,
      public SecretnessAnalysisDependent<DimensionAnalysis> {
 public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<DimensionAnalysis>;

  void setToExitState(DimensionLattice* lattice) override;

  LogicalResult visitOperation(
      Operation* op, ArrayRef<DimensionLattice*> operands,
      ArrayRef<const DimensionLattice*> results) override;

  // dummy impl
  void visitBranchOperand(OpOperand& operand) override {}
  void visitCallOperand(OpOperand& operand) override {}
};

std::optional<DimensionState::DimensionType> getDimension(
    Value value, DataFlowSolver* solver);

DimensionState::DimensionType getDimensionFromMgmtAttr(Value value);

constexpr StringRef kArgDimensionAttrName = "mgmt.dimension";

void annotateDimension(Operation* top, DataFlowSolver* solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_DIMENSIONANALYSIS_DIMENSIONANALYSIS_H_
