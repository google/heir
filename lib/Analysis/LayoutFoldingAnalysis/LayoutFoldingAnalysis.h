#ifndef LIB_ANALYSIS_LAYOUTFOLDINGANALYSIS_LAYOUTFOLDINGANALYSIS_H_
#define LIB_ANALYSIS_LAYOUTFOLDINGANALYSIS_LAYOUTFOLDINGANALYSIS_H_

#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// LayoutFoldingAnalysis provides a means to determine if the layout of a given
// SSA value is free, i.e., can be changed without additional cost of repacking.
// This happens when:
//
// - The SSA value is a function argument of a main entry point function,
//   so that its packing is determined by the compiler at provided as a helper
//   function that the client is assumed to call to pack/encrypt their inputs.
// - The SSA value is a plaintext value whose layout is assigned by the
// compiler.
// - The SSA value is downstream of any of the above, and only passing through
//   operations that are known to be layout-invariant (e.g., an elementwise
//   add).

// A simple dataclass describing if an SSA value may change its layout for free.
class LayoutIsFree {
 public:
  LayoutIsFree() : isFree(false) {}
  explicit LayoutIsFree(bool value) : isFree(value) {}
  ~LayoutIsFree() = default;

  bool operator==(const LayoutIsFree& rhs) const {
    return isFree == rhs.isFree;
  }

  bool getValue() const { return isFree; }

  static LayoutIsFree join(const LayoutIsFree& lhs, const LayoutIsFree& rhs) {
    // If either layout is not free, then the result is not free.
    return LayoutIsFree(lhs.getValue() && rhs.getValue());
  }

  static LayoutIsFree combine(llvm::ArrayRef<LayoutIsFree> lifs) {
    LayoutIsFree result = LayoutIsFree(true);
    for (const auto& lif : lifs) {
      if (!lif.getValue()) return LayoutIsFree(false);
    }
    return result;
  }

  void print(raw_ostream& os) const { os << "LayoutIsFree: " << isFree; }

 private:
  bool isFree;
};

raw_ostream& operator<<(raw_ostream& os, const LayoutIsFree& value);

class LayoutIsFreeLattice : public dataflow::Lattice<LayoutIsFree> {
 public:
  using Lattice::Lattice;
};

// An analysis that identifies and assigns the Secretness of an SSA value in a
// program. This is used by other passes to selectively apply transformations on
// operations that evaluate secret types. We use a forward dataflow analysis
// because the Secretness state propagates forward from the input arguments of a
// function down to values produced from operations that use these arguments.

class LayoutIsFreeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<LayoutIsFreeLattice> {
 public:
  explicit LayoutIsFreeAnalysis(DataFlowSolver& solver)
      : SparseForwardDataFlowAnalysis(solver) {}
  ~LayoutIsFreeAnalysis() override = default;
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(LayoutIsFreeLattice* lattice) override;

  LogicalResult visitOperation(Operation* operation,
                               ArrayRef<const LayoutIsFreeLattice*> operands,
                               ArrayRef<LayoutIsFreeLattice*> results) override;

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const LayoutIsFreeLattice*> argumentLattices,
      ArrayRef<LayoutIsFreeLattice*> resultLattices) override;
};

// Returns true if the layout of the given value is free.
bool isLayoutFree(Value value, DataFlowSolver* solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_LAYOUTFOLDINGANALYSIS_LAYOUTFOLDINGANALYSIS_H_
