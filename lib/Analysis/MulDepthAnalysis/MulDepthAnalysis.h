#ifndef LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
#define LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <variant>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/Utils.h"
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
  struct Uninit {
    bool operator==(const Uninit&) const = default;
  };
  struct Invalid {
    bool operator==(const Invalid&) const = default;
  };

  using MulDepthType = std::variant<Uninit, Invalid, int64_t>;

  MulDepthState() : value(Uninit{}) {}
  explicit MulDepthState(MulDepthType value) : value(value) {}
  MulDepthState(int64_t depth) : value(depth) {}
  ~MulDepthState() = default;

  int64_t getMulDepth() const {
    assert(isInt());
    return std::get<int64_t>(value);
  }

  void setMulDepth(int64_t depth) { value = depth; }

  bool operator==(const MulDepthState& rhs) const { return value == rhs.value; }

  bool isInitialized() const { return !std::holds_alternative<Uninit>(value); }

  bool isInt() const { return std::holds_alternative<int64_t>(value); }
  bool isInvalid() const { return std::holds_alternative<Invalid>(value); }

  static MulDepthState join(const MulDepthState& lhs,
                            const MulDepthState& rhs) {
    return std::visit(Overloaded{
                          [](Invalid, auto) -> MulDepthState {
                            return MulDepthState(Invalid{});
                          },
                          [](Uninit, auto other) -> MulDepthState {
                            return MulDepthState(other);
                          },
                          [](int64_t, Invalid) -> MulDepthState {
                            return MulDepthState(Invalid{});
                          },
                          [](int64_t lhsVal, Uninit) -> MulDepthState {
                            return MulDepthState(lhsVal);
                          },
                          [](int64_t lhsVal, int64_t rhsVal) -> MulDepthState {
                            return MulDepthState(std::max(lhsVal, rhsVal));
                          },
                      },
                      lhs.value, rhs.value);
  }

  void print(llvm::raw_ostream& os) const {
    std::visit(
        Overloaded{[&](Uninit) { os << "MulDepthState(uninitialized)"; },
                   [&](Invalid) { os << "MulDepthState(Invalid)"; },
                   [&](int64_t val) { os << "MulDepthState(" << val << ")"; }},
        value);
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const MulDepthState& state) {
    state.print(os);
    return os;
  }

 private:
  MulDepthType value;
};

class MulDepthLattice : public dataflow::Lattice<MulDepthState> {
 public:
  using Lattice::Lattice;
};

class MulDepthAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<MulDepthLattice>,
      public SecretnessAnalysisDependent<MulDepthAnalysis> {
 public:
  MulDepthAnalysis(DataFlowSolver& solver, int mulDepthBudget = 40)
      : dataflow::SparseForwardDataFlowAnalysis<MulDepthLattice>(solver),
        mulDepthBudget(mulDepthBudget) {}
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

 private:
  int mulDepthBudget;
};

MulDepthState deriveResultMulDepth(Operation* op,
                                   ArrayRef<const MulDepthLattice*> operands);

int64_t getMaxMulDepth(Operation* op, DataFlowSolver& solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
