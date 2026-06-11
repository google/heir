#ifndef LIB_ANALYSIS_PREPROCESSINGANALYSIS_PREPROCESSINGANALYSIS_H_
#define LIB_ANALYSIS_PREPROCESSINGANALYSIS_PREPROCESSINGANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// A lattice element for the preprocessing analysis. It stores a sorted set of
/// Operation pointers corresponding to the set of downstream encode ops that
/// depend on a given Operation.
class PreprocessingState {
 public:
  PreprocessingState() : initialized(false) {}
  explicit PreprocessingState(bool initialized) : initialized(initialized) {}

  explicit PreprocessingState(Operation* op) : initialized(true) {
    if (op) {
      encodeOps.push_back(op);
    }
  }

  PreprocessingState(SmallVector<Operation*, 4> ops, bool init)
      : encodeOps(std::move(ops)), initialized(init) {
    if (!encodeOps.empty()) {
      llvm::sort(encodeOps);
      encodeOps.erase(std::unique(encodeOps.begin(), encodeOps.end()),
                      encodeOps.end());
    }
  }

  bool isInitialized() const { return initialized; }

  ArrayRef<Operation*> getEncodeOps() const {
    assert(isInitialized());
    return encodeOps;
  }

  bool operator==(const PreprocessingState& rhs) const {
    if (initialized != rhs.initialized) return false;
    if (!initialized) return true;
    return encodeOps == rhs.encodeOps;
  }

  // Meet two states by taking a union over their underlying sets
  // of pointers. Use meet instead of join since it's a backwards
  // analysis.
  static PreprocessingState meet(const PreprocessingState& lhs,
                                 const PreprocessingState& rhs) {
    if (lhs == rhs) return lhs;
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    SmallVector<Operation*, 4> joined;
    joined.reserve(std::max(lhs.encodeOps.size(), rhs.encodeOps.size()));
    std::set_union(lhs.encodeOps.begin(), lhs.encodeOps.end(),
                   rhs.encodeOps.begin(), rhs.encodeOps.end(),
                   std::back_inserter(joined));
    return PreprocessingState(std::move(joined), true);
  }

  // Need a dummy implementation of join to satisfy Lattice<T>.
  static PreprocessingState join(const PreprocessingState& lhs,
                                 const PreprocessingState& rhs) {
    return meet(lhs, rhs);
  }

  void insert(Operation* op) {
    initialized = true;
    assert(op != nullptr);
    auto it = llvm::lower_bound(encodeOps, op);
    if (it == encodeOps.end() || *it != op) {
      encodeOps.insert(it, op);
    }
  }

  void print(llvm::raw_ostream& os) const {
    if (!isInitialized()) {
      os << "PreprocessingState(uninitialized)";
      return;
    }
    os << "PreprocessingState([";
    bool first = true;
    for (auto* op : encodeOps) {
      if (!first) os << ", ";
      first = false;
      if (op) os << op->getName();
    }
    os << "])";
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const PreprocessingState& state) {
    state.print(os);
    return os;
  }

 private:
  // A (sorted, unique) list of encode ops.
  SmallVector<Operation*, 4> encodeOps;
  bool initialized;
};

class PreprocessingLattice : public dataflow::Lattice<PreprocessingState> {
 public:
  using Lattice::Lattice;
};

class PreprocessingAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<PreprocessingLattice> {
 public:
  explicit PreprocessingAnalysis(DataFlowSolver& solver,
                                 SymbolTableCollection& symbolTable)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable) {}

  void setToExitState(PreprocessingLattice* lattice) override;

  LogicalResult visitOperation(
      Operation* op, ArrayRef<PreprocessingLattice*> operands,
      ArrayRef<const PreprocessingLattice*> results) override;

  void visitBranchOperand(OpOperand& operand) override;
  void visitCallOperand(OpOperand& operand) override {}

  void visitNonControlFlowArguments(RegionSuccessor& successor,
                                    ArrayRef<BlockArgument> arguments) override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_PREPROCESSINGANALYSIS_PREPROCESSINGANALYSIS_H_
