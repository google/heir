#ifndef LIB_ANALYSIS_LEVELANALYSIS_BOOTSTRAPWATERLINEANALYSIS_H_
#define LIB_ANALYSIS_LEVELANALYSIS_BOOTSTRAPWATERLINEANALYSIS_H_

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

class BootstrapWaterlineState {
 public:
  BootstrapWaterlineState() : levelState(Uninit{}), needsBootstrap(false) {}
  BootstrapWaterlineState(LevelState levelState, bool needsBootstrap)
      : levelState(levelState), needsBootstrap(needsBootstrap) {}

  LevelState getLevelState() const { return levelState; }
  bool getNeedsBootstrap() const { return needsBootstrap; }

  bool operator==(const BootstrapWaterlineState& other) const {
    return levelState == other.levelState &&
           needsBootstrap == other.needsBootstrap;
  }

  static BootstrapWaterlineState join(const BootstrapWaterlineState& lhs,
                                      const BootstrapWaterlineState& rhs) {
    return BootstrapWaterlineState(
        LevelState::join(lhs.levelState, rhs.levelState),
        lhs.needsBootstrap || rhs.needsBootstrap);
  }

  void print(llvm::raw_ostream& os) const {
    os << "BWState(";
    levelState.print(os);
    os << ", needsBootstrap=" << (needsBootstrap ? "true" : "false") << ")";
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const BootstrapWaterlineState& state) {
    state.print(os);
    return os;
  }

 private:
  LevelState levelState;
  bool needsBootstrap;
};

class BootstrapWaterlineLattice
    : public dataflow::Lattice<BootstrapWaterlineState> {
 public:
  using Lattice::Lattice;
};

class BootstrapWaterlineAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<BootstrapWaterlineLattice>,
      public SecretnessAnalysisDependent<BootstrapWaterlineAnalysis> {
 public:
  BootstrapWaterlineAnalysis(DataFlowSolver& solver, int waterline = 20,
                             int levelBudget = 20)
      : dataflow::SparseForwardDataFlowAnalysis<BootstrapWaterlineLattice>(
            solver),
        waterline(waterline),
        levelBudget(levelBudget) {}
  friend class SecretnessAnalysisDependent<BootstrapWaterlineAnalysis>;

  void setToEntryState(BootstrapWaterlineLattice* lattice) override {
    propagateIfChanged(
        lattice, lattice->join(BootstrapWaterlineState(LevelState(0), false)));
  }

  LogicalResult visitOperation(
      Operation* op, ArrayRef<const BootstrapWaterlineLattice*> operands,
      ArrayRef<BootstrapWaterlineLattice*> results) override;

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const BootstrapWaterlineLattice*> argumentLattices,
      ArrayRef<BootstrapWaterlineLattice*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }

 private:
  int waterline;
  int levelBudget;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_LEVELANALYSIS_BOOTSTRAPWATERLINEANALYSIS_H_
