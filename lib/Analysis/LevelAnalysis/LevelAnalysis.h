#ifndef LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_
#define LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>
#include <variant>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

// This file contains a pair of forward and backward analyses that allow one to
// determine the "level" (in the sense of remaining multiplicative depth) of SSA
// values in a possibly-loop-unrolled program.
//
// It defines a lattice (LevelLattice) and combines a forward analysis for
// determining the result level of operations that operate on secrets, along
// with a backward analysis that deduces the required level of plaintext
// operands of ciphertext-plaintext ops, which can be used to determine how
// to encode a cleartext as a plaintext.
//
// Because this analysis is used to decide what the total number of levels will
// ultimately be in a compiled program, it cannot start from some "initial"
// level and go down to zero as operations are visited. Instead, we have a
// (somewhat confusing) internal convention where the "level" starts at 0
// and increases as "mod_reduce" or "level_reduce" operations are visited,
// and then at the end when the level data is materialized (see helpers like
// annotateLevel and getMaxLevel), this 0, ..., MaxLevel is reversed to have
// the level of an initial ciphertext be equal to the MaxLevel found in the
// analysis, and mod_reduce/level_reduce causes levels do decrement.
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

// An element of a linear lattice, whose elements are:
//
// Uninint < 0 < 1 < ... all positive integers ... < MaxLevel < Invalid
//
// In particular, the join computes the max and the meet computes the min, with
// special cases for the sentinel values on the ends.
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

  bool isInt() const { return std::holds_alternative<int>(value); }

  bool isMaxLevel() const { return std::holds_alternative<MaxLevel>(value); }

  int64_t getInt() const { return std::get<int>(value); }

  // Join goes "up" in the lattice.
  static LevelState join(const LevelState& lhs, const LevelState& rhs) {
    return std::visit(
        Overloaded{
            [](Invalid, auto) -> LevelState { return LevelState(Invalid{}); },
            [](Uninit, auto other) -> LevelState { return LevelState(other); },
            [](MaxLevel, Invalid) -> LevelState {
              return LevelState(Invalid{});
            },
            [](MaxLevel, Uninit) -> LevelState {
              return LevelState(MaxLevel{});
            },
            [](MaxLevel, MaxLevel) -> LevelState {
              return LevelState(MaxLevel{});
            },
            [](MaxLevel, int) -> LevelState { return LevelState(MaxLevel{}); },
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

/// Forward Analyze the level of each secret Value
///
/// Note that the value stored in LevelState is from 0 to L instead of the
/// L-to-0 FHE convention. This is because at the beginning we have no
/// information of L and we rely on this analysis to get L (getMaxLevel). In
/// annotateLevel we will convert the level to L - level.
///
/// Summary of op transfer functions
///
/// - mod_reduce: increase level by 1
/// - bootstrap: set the level back to 0 (input level)
/// - level_reduce: increase the level by a value determined by the op
/// - level_reduce_min op: set the level to a sentinel "max" value (unknown at
///   the start of the analysis).
///
/// This analysis is expected to determine all the levels of the secret Value,
/// or ciphertext in the program. The level of plaintext Values should be
/// determined by the Backward Analysis below.
class LevelAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<LevelLattice>,
      public SecretnessAnalysisDependent<LevelAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<LevelAnalysis>;

  void setToEntryState(LevelLattice* lattice) override {
    propagateIfChanged(lattice, lattice->join(LevelState(0)));
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

/// Backward Analyze the level of plaintext operands of ct-pt ops.
///
/// This analysis should be run after the (forward) LevelAnalysis where the
/// level of all the secret Value is determined. Then, this analysis will find
/// ct-pt pair and determine the level of the pt Value.
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

// Get the maximum level of SSA values in the op, from the data flow solver.
int getMaxLevel(Operation* top, DataFlowSolver* solver);

LevelState transferForward(mgmt::ModReduceOp op,
                           ArrayRef<const LevelLattice*> operands);
LevelState transferForward(mgmt::LevelReduceOp op,
                           ArrayRef<const LevelLattice*> operands);
LevelState transferForward(mgmt::LevelReduceMinOp op,
                           ArrayRef<const LevelLattice*> operands);
LevelState transferForward(mgmt::BootstrapOp op,
                           ArrayRef<const LevelLattice*> operands);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_LEVELANALYSIS_LEVELANALYSIS_H_
