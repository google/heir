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

class RangeState {
 public:
  /// Represents a range using Log2Arithmetic.
  /// Here we only store the bound.
  /// For [a, b], store Log2Arithmetic::of(max(abs(a), abs(b))).
  using RangeType = Log2Arithmetic;

  RangeState() : range(std::nullopt) {}
  explicit RangeState(RangeType range) : range(range) {}
  ~RangeState() = default;

  RangeType getRange() const {
    assert(isInitialized());
    return range.value();
  }
  RangeType get() const { return getRange(); }

  bool operator==(const RangeState &rhs) const { return range == rhs.range; }

  bool isInitialized() const { return range.has_value(); }

  static RangeState join(const RangeState &lhs, const RangeState &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return RangeState{std::max(lhs.getRange(), rhs.getRange())};
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "RangeState(normal: "
         << doubleToString2Prec(range.value().getValue())
         << ", log2: " << doubleToString2Prec(range.value().getLog2Value())
         << ")";
    } else {
      os << "RangeState(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const RangeState &state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<RangeType> range;
};

class RangeLattice : public dataflow::Lattice<RangeState> {
 public:
  using Lattice::Lattice;
};

class RangeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<RangeLattice>,
      public SecretnessAnalysisDependent<RangeAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<RangeAnalysis>;

  RangeAnalysis(DataFlowSolver &solver, Log2Arithmetic inputRange)
      : dataflow::SparseForwardDataFlowAnalysis<RangeLattice>(solver),
        inputRange(inputRange) {}

  void setToEntryState(RangeLattice *lattice) override {
    // This handles both secret input and plaintext func arg
    propagateIfChanged(lattice, lattice->join(RangeState({inputRange})));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const RangeLattice *> operands,
                               ArrayRef<RangeLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const RangeLattice *> argumentLattices,
                         ArrayRef<RangeLattice *> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState *state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }

 private:
  Log2Arithmetic inputRange;
};

std::optional<RangeState::RangeType> getRange(Value value,
                                              DataFlowSolver *solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_CKKSRANGEANALYSIS_CKKSRANGEANALYSIS_H_
