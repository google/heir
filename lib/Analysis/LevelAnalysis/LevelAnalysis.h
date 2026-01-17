#ifndef LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_
#define LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>
#include <variant>

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

// This analysis should be used after --mlir-to-secret-arithmetic
// but before --secret-distribute-generic
// where a whole secret::GenericOp is assumed
namespace mlir {
namespace heir {

// A sentinel for the maximum allowable level before it is determined exactly
// what the max level is. In the semantics of this analysis, levels start from 0
// (the initial level) and go up to some max level (determined by a pass that
// uses this analysis). Some ops, such as reduce_level_min, jump straight to
// the max level, and it may not yet be determined what that max level is.
// At the end of the analysis, when the levels are converted from 0, ..., Max
// to Max, ..., 0, the sentinel is "resolved" to a concrete max level.
struct MaxLevel {
  bool operator==(const MaxLevel&) const = default;
};

// A sentinel for an uninitialized level
struct Uninit {
  bool operator==(const Uninit&) const = default;
};

// A sentinel for an invalid level (e.g., mod_reduce on a MaxLevel or Uninit)
struct Invalid {
  bool operator==(const Invalid&) const = default;
};

// Helper for the "overloaded" pattern
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

class LevelState {
 public:
  using LevelType = std::variant<MaxLevel, Uninit, Invalid, int>;

  LevelState() : value(Uninit{}) {}
  explicit LevelState(LevelType level) : value(level) {}
  LevelState(int level) : value(level) {}
  LevelState(int64_t level) : value((int)level) {}
  ~LevelState() = default;

  LevelType getLevel() const { return value; }
  void setLevel(LevelType val) { value = val; }
  LevelType get() const { return getLevel(); }

  bool operator==(const LevelState& other) const = default;

  bool isInitialized() const { return !std::holds_alternative<Uninit>(value); }

  bool isMaxLevel() const { return std::holds_alternative<MaxLevel>(value); }

  int64_t getInt() const { return std::get<int>(value); }

  static LevelState join(const LevelState& lhs, const LevelState& rhs) {
    return std::visit(
        Overloaded{
            [](MaxLevel, auto) -> LevelState { return LevelState(MaxLevel{}); },
            [](Uninit, auto other) -> LevelState { return LevelState(other); },
            [](Invalid, auto) -> LevelState { return LevelState(Invalid{}); },
            [](int, MaxLevel) -> LevelState { return LevelState(MaxLevel{}); },
            [](int lhsVal, Uninit) -> LevelState { return LevelState(lhsVal); },
            [](int, Invalid) -> LevelState { return LevelState(Invalid{}); },
            [](int lhsVal, int rhsVal) -> LevelState {
              return LevelState(std::max(lhsVal, rhsVal));
            },
        },
        lhs.value, rhs.value);
  }

  void print(llvm::raw_ostream& os) const {
    std::visit(Overloaded{[&](MaxLevel) { os << "Level(Max)"; },
                          [&](Uninit) { os << "Level(Uninitialized)"; },
                          [&](Invalid) { os << "Level(Invalid)"; },
                          [&](int val) { os << "Level(" << val << ")"; }},
               value);
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const LevelState& state) {
    state.print(os);
    return os;
  }

 private:
  LevelType value;
};

class LevelLattice : public dataflow::Lattice<LevelState> {
 public:
  using Lattice::Lattice;
};

/// Forward Analyse the level of each secret Value
///
/// Note that the value stored in LevelState is from 0 to L
/// instead of the L-to-0 FHE convention. This is because at
/// the beginning we have no information of L and we rely on
/// this analysis to get L (getMaxLevel). In annotateLevel
/// we will convert the level to L - level.
///
/// This forward analysis roots from user input as 0, and
/// after each modulus switching operation, the level will
/// increase by 1, until the output.
///
/// Special case is the bootstrapping operation, where the level
/// will be set back to 0 (input level).
///
/// This analysis is expected to determine all the levels of
/// the secret Value, or ciphertext in the program.
/// The level of plaintext Value should be determined by the
/// Backward Analysis below.
class LevelAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<LevelLattice>,
      public SecretnessAnalysisDependent<LevelAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<LevelAnalysis>;

  void setToEntryState(LevelLattice* lattice) override {
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice, lattice->join(LevelState(0)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(LevelState()));
  }

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const LevelLattice*> operands,
                               ArrayRef<LevelLattice*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const LevelLattice*> argumentLattices,
                         ArrayRef<LevelLattice*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

LevelState deriveResultLevel(Operation* op,
                             ArrayRef<const LevelLattice*> operands);

/// Backward Analyse the level of plaintext Value
///
/// This analysis should be run after the (forward) LevelAnalysis
/// where the level of all the secret Value is determined.
/// Then, this analysis will find ct-pt pair and determine the
/// level of the pt Value.
class LevelAnalysisBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<LevelLattice>,
      public SecretnessAnalysisDependent<LevelAnalysis> {
 public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<LevelAnalysis>;

  void setToExitState(LevelLattice* lattice) override {
    propagateIfChanged(lattice, lattice->join(LevelState()));
  }

  LogicalResult visitOperation(Operation* op, ArrayRef<LevelLattice*> operands,
                               ArrayRef<const LevelLattice*> results) override;

  // dummy impl
  void visitBranchOperand(OpOperand& operand) override {}
  void visitCallOperand(OpOperand& operand) override {}
  void visitNonControlFlowArguments(
      RegionSuccessor& successor, ArrayRef<BlockArgument> arguments) override {}
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

LevelState getLevelFromMgmtAttr(Value value);

constexpr StringRef kArgLevelAttrName = "mgmt.level";

/// baseLevel is for B/FV scheme, where all the analysis result would be 0
void annotateLevel(Operation* top, DataFlowSolver* solver, int baseLevel = 0);

// Get the maximum annotated level from mgmt attributes.
// Assumes max level at the entrypoint to the main compiled function.
std::optional<int> getMaxLevel(Operation* root);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_
