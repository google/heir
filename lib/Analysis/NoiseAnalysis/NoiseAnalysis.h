#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project

namespace mlir {
namespace heir {

/// This lattice element represents the noise distribution of an SSA value.
class NoiseLattice : public dataflow::Lattice<Noise> {
 public:
  using Lattice::Lattice;
};

class NoiseAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<NoiseLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  NoiseAnalysis(DataFlowSolver &solver, const SchemeParam &schemeParam)
      : dataflow::SparseForwardDataFlowAnalysis<NoiseLattice>(solver),
        schemeParam(schemeParam) {}

  void setToEntryState(NoiseLattice *lattice) override {
    // At an entry point, we have no information about the noise.
    propagateIfChanged(lattice, lattice->join(Noise::uninitialized()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const NoiseLattice *> operands,
                               ArrayRef<NoiseLattice *> results) override;

 private:
  const SchemeParam schemeParam;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
