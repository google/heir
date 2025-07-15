#ifndef LIB_ANALYSIS_CKKSRANGEANALYSIS_CKKSRANGEANALYSIS_H_
#define LIB_ANALYSIS_CKKSRANGEANALYSIS_CKKSRANGEANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Utils/LogArithmetic.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

class CKKSRangeState {
 public:
  /// Represents a CKKS range using Log2Arithmetic.
  /// Here we only store the bound.
  /// For double input in range [-1, 1], we use Log2Arithmetic::of(1) to
  /// represent it.
  using CKKSRangeType = Log2Arithmetic;

  CKKSRangeState() : range(std::nullopt) {}
  explicit CKKSRangeState(CKKSRangeType range) : range(range) {}
  ~CKKSRangeState() = default;

  CKKSRangeType getCKKSRange() const {
    assert(isInitialized());
    return range.value();
  }
  CKKSRangeType get() const { return getCKKSRange(); }

  bool operator==(const CKKSRangeState &rhs) const {
    return range == rhs.range;
  }

  bool isInitialized() const { return range.has_value(); }

  static CKKSRangeState join(const CKKSRangeState &lhs,
                             const CKKSRangeState &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return CKKSRangeState{std::max(lhs.getCKKSRange(), rhs.getCKKSRange())};
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "CKKSRangeState(normal: "
         << doubleToString2Prec(range.value().getValue())
         << ", log2: " << doubleToString2Prec(range.value().getLog2Value())
         << ")";
    } else {
      os << "CKKSRangeState(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const CKKSRangeState &state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<CKKSRangeType> range;
};

class CKKSRangeLattice : public dataflow::Lattice<CKKSRangeState> {
 public:
  using Lattice::Lattice;
};

class CKKSRangeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<CKKSRangeLattice>,
      public SecretnessAnalysisDependent<CKKSRangeAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<CKKSRangeAnalysis>;

  void setToEntryState(CKKSRangeLattice *lattice) override {
    // For double input, default range is [-1, 1]
    // This handles both secret input and plaintext func arg
    propagateIfChanged(lattice,
                       lattice->join(CKKSRangeState({Log2Arithmetic::of(1)})));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const CKKSRangeLattice *> operands,
                               ArrayRef<CKKSRangeLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const CKKSRangeLattice *> argumentLattices,
                         ArrayRef<CKKSRangeLattice *> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState *state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

std::optional<CKKSRangeState::CKKSRangeType> getCKKSRange(
    Value value, DataFlowSolver *solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_CKKSRANGEANALYSIS_CKKSRANGEANALYSIS_H_
