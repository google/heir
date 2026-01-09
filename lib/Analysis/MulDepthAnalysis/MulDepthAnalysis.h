#ifndef LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
#define LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// This analysis should be used after --mlir-to-secret-arithmetic
// but before --secret-distribute-generic
// where a whole secret::GenericOp is assumed

// represent the maximal depth of multiplication used to produce the Value,
// where multiplication is in secretness domain
class MulDepthState {
 public:
  MulDepthState() : mulDepth(std::nullopt) {}
  explicit MulDepthState(int64_t mulDepth) : mulDepth(mulDepth) {}
  ~MulDepthState() = default;

  int64_t getMulDepth() const {
    assert(isInitialized());
    return mulDepth.value();
  }

  void setMulDepth(int64_t depth) {
    mulDepth = std::make_optional<int64_t>(depth);
  }

  bool operator==(const MulDepthState& rhs) const {
    return mulDepth == rhs.mulDepth;
  }

  bool isInitialized() const { return mulDepth.has_value(); }

  static MulDepthState join(const MulDepthState& lhs,
                            const MulDepthState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return MulDepthState{std::max(lhs.getMulDepth(), rhs.getMulDepth())};
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "MulDepthState(" << mulDepth.value() << ")";
    } else {
      os << "MulDepthState(uninitialized)";
    }
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const MulDepthState& state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<int64_t> mulDepth;
};

class MulDepthLattice : public dataflow::Lattice<MulDepthState> {
 public:
  using Lattice::Lattice;
};

class MulDepthAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<MulDepthLattice>,
      public SecretnessAnalysisDependent<MulDepthAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<MulDepthAnalysis>;

  void setToEntryState(MulDepthLattice* lattice) override {
    if (isa<SecretTypeInterface>(
            getElementTypeOrSelf(lattice->getAnchor().getType()))) {
      propagateIfChanged(lattice, lattice->join(MulDepthState(0)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(MulDepthState()));
  }

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const MulDepthLattice*> operands,
                               ArrayRef<MulDepthLattice*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const MulDepthLattice*> argumentLattices,
                         ArrayRef<MulDepthLattice*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

FailureOr<int64_t> deriveResultMulDepth(
    Operation* op, ArrayRef<const MulDepthLattice*> operands);

int64_t getMaxMulDepth(Operation* op, DataFlowSolver& solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
