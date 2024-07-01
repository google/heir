#ifndef LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
#define LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_

#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace mul_depth_analysis {

class MulDepth {
 public:
  MulDepth() : value(std::nullopt) {}
  MulDepth(int64_t value) : value(value) {}
  ~MulDepth() = default;

  bool isInitialized() const { return value.has_value(); }

  const int64_t &getValue() const {
    assert(isInitialized());
    return *value;
  }

  bool operator==(const MulDepth &rhs) const { return value == rhs.value; }

  static MulDepth join(const MulDepth &lhs, const MulDepth &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    return MulDepth{lhs.getValue() > rhs.getValue() ? lhs.getValue()
                                                    : rhs.getValue()};
  }

  void print(raw_ostream &os) const { os << value; }

 private:
  std::optional<int64_t> value;

  friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diagnostic,
                                      const MulDepth &foo) {
    if (foo.isInitialized()) {
      return diagnostic << foo.getValue();
    }
    return diagnostic << "uninitialized";
  }
};

inline raw_ostream &operator<<(raw_ostream &os, const MulDepth &v) {
  v.print(os);
  return os;
}

class MulDepthLattice : public dataflow::Lattice<MulDepth> {
 public:
  using Lattice::Lattice;
};

class MulDepthAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<MulDepthLattice> {
 public:
  explicit MulDepthAnalysis(DataFlowSolver &solver,
                            const DataFlowSolver *sccpAnalysis)
      : SparseForwardDataFlowAnalysis(solver), sccpAnalysis(sccpAnalysis) {}
  ~MulDepthAnalysis() override = default;
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<const MulDepthLattice *> operands,
                      ArrayRef<MulDepthLattice *> results) override;

  void setToEntryState(MulDepthLattice *lattice) override;

 private:
  const DataFlowSolver *sccpAnalysis;
};

}  // namespace mul_depth_analysis
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_MULDEPTHANALYSIS_MULDEPTHANALYSIS_H_
