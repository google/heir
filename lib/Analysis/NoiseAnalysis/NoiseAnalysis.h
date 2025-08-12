#ifndef LIB_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
#define LIB_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

/// This lattice element represents the noise data of an SSA value.
template <typename NoiseState>
class NoiseLattice : public dataflow::Lattice<NoiseState> {
 public:
  using dataflow::Lattice<NoiseState>::Lattice;
};

/// This analysis template takes a noise model as argument and computes the
/// noise data for each SSA value in the program. The exact instantiations of
/// member functions are dependent on the noise model hence explicit
/// specialization is required for each noise model.
template <typename NoiseModelT>
class NoiseAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          NoiseLattice<typename NoiseModelT::StateType>>,
      public SecretnessAnalysisDependent<NoiseAnalysis<NoiseModelT>> {
 public:
  friend class SecretnessAnalysisDependent<NoiseAnalysis<NoiseModelT>>;

  using NoiseModel = NoiseModelT;
  using NoiseState = typename NoiseModelT::StateType;
  using LatticeType = NoiseLattice<NoiseState>;
  using SchemeParamType = typename NoiseModelT::SchemeParamType;
  using LocalParamType = typename NoiseModelT::LocalParamType;

  using dataflow::SparseForwardDataFlowAnalysis<
      LatticeType>::SparseForwardDataFlowAnalysis;

  NoiseAnalysis(DataFlowSolver& solver, const SchemeParamType& schemeParam,
                const NoiseModelT& noiseModel)
      : dataflow::SparseForwardDataFlowAnalysis<LatticeType>(solver),
        schemeParam(schemeParam),
        noiseModel(noiseModel) {}

  void setToEntryState(LatticeType* lattice) override;

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const LatticeType*> operands,
                               ArrayRef<LatticeType*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const LatticeType*> argumentLattices,
                         ArrayRef<LatticeType*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    this->propagateIfChanged(state, changed);
  }

 private:
  const SchemeParamType schemeParam;
  const NoiseModelT& noiseModel;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
