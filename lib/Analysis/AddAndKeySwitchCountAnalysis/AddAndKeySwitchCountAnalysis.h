#ifndef LIB_ANALYSIS_ADDANDKEYSWITCHCOUNTANALYSISANALYSIS_ADDANDKEYSWITCHCOUNTANALYSISANALYSIS_H_
#define LIB_ANALYSIS_ADDANDKEYSWITCHCOUNTANALYSISANALYSIS_ADDANDKEYSWITCHCOUNTANALYSISANALYSIS_H_

#include <algorithm>
#include <cassert>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// This analysis should be used after --secret-with-mgmt-bgv
// but before --secret-distribute-generic
// where a whole secret::GenericOp is assumed
//
// this follows strictly the strategy of modulus switch
// right before multiplication including the first.
// namely FLEXIBLEAUTOEXT in OpenFHE
//
// OpenFHE only supports setting EvalAddCount/EvalKeySwitchCount
// for BGV/BFV, so HEIR only supports BGV here
//
// For not include-the-first, the addCount for the L-th level
// might be overestimated, as we can not distinguish between
// Vmult and Vfresh

class CountState {
 public:
  CountState() : initialized(false), addCount(0), keySwitchCount(0) {}
  explicit CountState(int addCount, int keySwitchCount)
      : initialized(true), addCount(addCount), keySwitchCount(keySwitchCount) {}
  ~CountState() = default;

  int getAddCount() const {
    assert(isInitialized());
    return addCount;
  }

  int getKeySwitchCount() const {
    assert(isInitialized());
    return keySwitchCount;
  }

  bool operator==(const CountState& rhs) const {
    return initialized == rhs.initialized && addCount == rhs.addCount &&
           keySwitchCount == rhs.keySwitchCount;
  }

  bool isInitialized() const { return initialized; }

  CountState operator+(const CountState& rhs) const {
    assert(isInitialized() && rhs.isInitialized());
    return CountState{addCount + rhs.addCount,
                      keySwitchCount + rhs.keySwitchCount};
  }

  CountState keySwitch() const {
    assert(isInitialized());
    return CountState{addCount, keySwitchCount + 1};
  }

  CountState max(const CountState& rhs) const {
    assert(isInitialized() && rhs.isInitialized());
    return CountState{std::max(addCount, rhs.addCount),
                      std::max(keySwitchCount, rhs.keySwitchCount)};
  }

  static CountState join(const CountState& lhs, const CountState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return lhs.max(rhs);
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "CountState(" << addCount << ", " << keySwitchCount << ")";
    } else {
      os << "CountState(uninitialized)";
    }
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const CountState& state) {
    state.print(os);
    return os;
  }

 private:
  bool initialized;
  int addCount;  // how many Vmult or Vfresh (before first multiplication)
                 // encountered
  int keySwitchCount;
};

class CountLattice : public dataflow::Lattice<CountState> {
 public:
  using Lattice::Lattice;
};

class CountAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<CountLattice>,
      public SecretnessAnalysisDependent<CountAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<CountAnalysis>;

  void setToEntryState(CountLattice* lattice) override {
    // one addition in Vfresh
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice, lattice->join(CountState(1, 0)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(CountState()));
  }

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const CountLattice*> operands,
                               ArrayRef<CountLattice*> results) override;
};

void annotateCount(Operation* top, DataFlowSolver* solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ADDANDKEYSWITCHCOUNTANALYSISANALYSIS_ADDANDKEYSWITCHCOUNTANALYSISANALYSIS_H_
