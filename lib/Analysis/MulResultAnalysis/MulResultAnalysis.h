#ifndef LIB_ANALYSIS_MULRESULTANALYSIS_MULRESULTANALYSIS_H_
#define LIB_ANALYSIS_MULRESULTANALYSIS_MULRESULTANALYSIS_H_

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

// represent whether a value is a multiplication result or the derived value of
// a multiplication result
// where mutiplication is in secretness domain
class MulResultState {
 public:
  MulResultState() : isMulResult(std::nullopt) {}
  explicit MulResultState(bool isMulResult) : isMulResult(isMulResult) {}
  ~MulResultState() = default;

  bool getIsMulResult() const {
    assert(isInitialized());
    return isMulResult.value();
  }

  bool operator==(const MulResultState &rhs) const {
    return isMulResult == rhs.isMulResult;
  }

  bool isInitialized() const { return isMulResult.has_value(); }

  static MulResultState join(const MulResultState &lhs,
                             const MulResultState &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return MulResultState{lhs.getIsMulResult() || rhs.getIsMulResult()};
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "MulResultState(" << isMulResult.value() << ")";
    } else {
      os << "MulResultState(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const MulResultState &state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<bool> isMulResult;
};

class MulResultLattice : public dataflow::Lattice<MulResultState> {
 public:
  using Lattice::Lattice;
};

class MulResultAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<MulResultLattice>,
      public SecretnessAnalysisDependent<MulResultAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<MulResultAnalysis>;

  void setToEntryState(MulResultLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(MulResultState()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const MulResultLattice *> operands,
                               ArrayRef<MulResultLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const MulResultLattice *> argumentLattices,
                         ArrayRef<MulResultLattice *> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState *state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_MULRESULTANALYSIS_MULRESULTANALYSIS_H_
