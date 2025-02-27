#ifndef LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_
#define LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
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

class LevelState {
 public:
  using LevelType = int;

  LevelState() : level(std::nullopt) {}
  explicit LevelState(LevelType level) : level(level) {}
  ~LevelState() = default;

  LevelType getLevel() const {
    assert(isInitialized());
    return level.value();
  }
  LevelType get() const { return getLevel(); }

  bool operator==(const LevelState &rhs) const { return level == rhs.level; }

  bool isInitialized() const { return level.has_value(); }

  static LevelState join(const LevelState &lhs, const LevelState &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return LevelState{std::max(lhs.getLevel(), rhs.getLevel())};
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "LevelState(" << level.value() << ")";
    } else {
      os << "LevelState(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const LevelState &state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<LevelType> level;
};

class LevelLattice : public dataflow::Lattice<LevelState> {
 public:
  using Lattice::Lattice;
};

class LevelAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<LevelLattice>,
      public SecretnessAnalysisDependent<LevelAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<LevelAnalysis>;

  void setToEntryState(LevelLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(LevelState()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const LevelLattice *> operands,
                               ArrayRef<LevelLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const LevelLattice *> argumentLattices,
                         ArrayRef<LevelLattice *> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState *state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

LevelState::LevelType getLevelFromMgmtAttr(Value value);

/// baseLevel is for B/FV scheme, where all the analysis result would be 0
void annotateLevel(Operation *top, DataFlowSolver *solver, int baseLevel = 0);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_
