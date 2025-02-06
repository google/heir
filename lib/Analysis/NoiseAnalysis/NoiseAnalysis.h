#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_

#include "lib/Analysis/NoiseAnalysis/BGV/Noise.h"
#include "lib/Analysis/NoiseAnalysis/Params.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// This lattice element represents the noise distribution of an SSA value.
template <typename Noise>
class NoiseLattice : public dataflow::Lattice<Noise> {
 public:
  using dataflow::Lattice<Noise>::Lattice;
};

template <typename Noise>
class NoiseAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<NoiseLattice<Noise>>,
      public SecretnessAnalysisDependent<NoiseAnalysis<Noise>> {
 public:
  using dataflow::SparseForwardDataFlowAnalysis<
      NoiseLattice<Noise>>::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<NoiseAnalysis<Noise>>;

  NoiseAnalysis(DataFlowSolver &solver, const SchemeParam &schemeParam)
      : dataflow::SparseForwardDataFlowAnalysis<NoiseLattice<Noise>>(solver),
        schemeParam(schemeParam) {}

  void setToEntryState(NoiseLattice<Noise> *lattice) override {
    // At an entry point, we have no information about the noise.
    this->propagateIfChanged(lattice, lattice->join(Noise::uninitialized()));
  }

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const NoiseLattice<Noise> *> operands,
      ArrayRef<NoiseLattice<Noise> *> results) override;

 private:
  const SchemeParam schemeParam;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
