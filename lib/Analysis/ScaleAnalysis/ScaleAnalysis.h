#ifndef LIB_ANALYSIS_MULRESULTANALYSIS_MULRESULTANALYSIS_H_
#define LIB_ANALYSIS_MULRESULTANALYSIS_MULRESULTANALYSIS_H_

#include <cassert>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Parameters/BGV/Params.h"
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

class ScaleState {
 public:
  ScaleState() : scale(std::nullopt) {}
  explicit ScaleState(int64_t scale) : scale(scale) {}
  ~ScaleState() = default;

  int64_t getScale() const {
    assert(isInitialized());
    return scale.value();
  }

  bool operator==(const ScaleState &rhs) const { return scale == rhs.scale; }

  bool isInitialized() const { return scale.has_value(); }

  static ScaleState join(const ScaleState &lhs, const ScaleState &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    // if both are initialized, they should be the same
    return lhs;
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "ScaleState(" << scale.value() << ")";
    } else {
      os << "ScaleState(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ScaleState &state) {
    state.print(os);
    return os;
  }

 private:
  // this may not represent 2 ** 80 scale for CKKS
  std::optional<int64_t> scale;
};

class ScaleLattice : public dataflow::Lattice<ScaleState> {
 public:
  using Lattice::Lattice;
};

class ScaleAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysis>;

  using SchemeParamType = typename bgv::SchemeParam;

  ScaleAnalysis(DataFlowSolver &solver, const SchemeParamType &schemeParam,
                int64_t inputScale)
      : dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>(solver),
        schemeParam(schemeParam),
        inputScale(inputScale) {}

  void setToEntryState(ScaleLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const ScaleLattice *> operands,
                               ArrayRef<ScaleLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const ScaleLattice *> argumentLattices,
                         ArrayRef<ScaleLattice *> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState *state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }

 private:
  const SchemeParamType schemeParam;
  int64_t inputScale;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_MULRESULTANALYSIS_MULRESULTANALYSIS_H_
